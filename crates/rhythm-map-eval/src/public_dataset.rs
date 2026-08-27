use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{self, Cursor, Read},
    net::{IpAddr, SocketAddr},
    path::{Component, Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result, anyhow, bail};
use flate2::read::DeflateDecoder;
use serde::{Deserialize, Serialize};
use ureq::{
    Agent,
    config::Config,
    http::Uri,
    unversioned::{
        resolver::{DefaultResolver, ResolvedSocketAddrs, Resolver},
        transport::{DefaultConnector, NextTimeout},
    },
};

use crate::dataset::sha256_file;

const DOWNLOAD_ATTEMPTS: usize = 3;
const ZIP_TAIL_BYTES: u64 = 1_048_576;
const MAX_CENTRAL_DIRECTORY_BYTES: u64 = 64 * 1_048_576;
const ZIP_RANGE_CHUNK_BYTES: u64 = 256 * 1_024;
const ZIP_MEMBER_RANGE_CONCURRENCY: usize = 8;

/// One file selected from a remotely hosted ZIP without downloading the archive.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PublicDatasetZipMember {
    /// Exact size of the containing ZIP, used to address and validate byte ranges.
    pub archive_size_bytes: u64,
    /// Exact case-sensitive member name in the upstream archive.
    pub member_path: String,
}

/// One remote ZIP member selected before its extracted byte identity is known.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicDatasetZipAssetSelection {
    /// Safe relative destination below the explicit output directory.
    pub path: String,
    /// Direct HTTPS URL of the containing ZIP.
    pub url: String,
    /// Whether the member is evaluation audio or an annotation source.
    pub role: PublicDatasetAssetRole,
    /// Immutable archive size and exact case-sensitive member path.
    pub zip_member: PublicDatasetZipMember,
}

/// Explicit TCP address for one HTTPS host during dataset acquisition.
///
/// The original URL remains unchanged, so HTTP Host, TLS SNI, and certificate
/// verification still use `host`. This is an operational escape hatch for a
/// broken local DNS resolver, not part of a dataset's immutable identity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicDatasetHostResolution {
    /// Exact case-insensitive hostname to override.
    pub host: String,
    /// TCP address used in place of DNS results for that hostname.
    pub address: IpAddr,
}

#[derive(Debug)]
struct AcquisitionResolver {
    overrides: HashMap<String, Vec<IpAddr>>,
    fallback: DefaultResolver,
}

impl Resolver for AcquisitionResolver {
    fn resolve(
        &self,
        uri: &Uri,
        config: &Config,
        timeout: NextTimeout,
    ) -> std::result::Result<ResolvedSocketAddrs, ureq::Error> {
        let Some(host) = uri.host() else {
            return self.fallback.resolve(uri, config, timeout);
        };
        let Some(addresses) = self.overrides.get(&host.to_ascii_lowercase()) else {
            return self.fallback.resolve(uri, config, timeout);
        };
        let port = uri.port_u16().unwrap_or_else(|| {
            if uri.scheme_str() == Some("https") {
                443
            } else {
                80
            }
        });
        let mut resolved = self.empty();
        for address in addresses {
            resolved.push(SocketAddr::new(*address, port));
        }
        Ok(resolved)
    }
}

/// Purpose of one immutable artifact in a public-dataset lock.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PublicDatasetAssetRole {
    /// Audio consumed by end-to-end evaluation.
    Audio,
    /// Upstream artifact retained to audit checked-in reference truth.
    AnnotationSource,
}

/// One content-addressed artifact distributed by a public dataset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PublicDatasetAsset {
    /// Safe relative destination below the explicit output directory.
    pub path: String,
    /// Direct HTTPS file URL, or the containing ZIP URL when `zip_member` is set.
    pub url: String,
    /// Lowercase SHA-256 of the exact downloaded bytes.
    pub sha256: String,
    /// Exact encoded file size.
    pub size_bytes: u64,
    /// Whether the artifact is evaluation audio or an annotation audit source.
    pub role: PublicDatasetAssetRole,
    /// Optional member to range-fetch from the ZIP at `url`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub zip_member: Option<PublicDatasetZipMember>,
}

/// Versioned source, rights, and immutable artifacts for a public dataset.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PublicDatasetLock {
    /// Lock schema version.
    pub schema_version: u32,
    /// Stable dataset identifier.
    pub id: String,
    /// Upstream dataset version represented by this lock.
    pub version: String,
    /// Canonical human-readable dataset page.
    pub homepage_url: String,
    /// SPDX identifier for the locked artifacts.
    pub license: String,
    /// Credit that must accompany redistributed or derived material.
    pub attribution: String,
    /// Immutable artifacts selected for this evaluation slice.
    pub assets: Vec<PublicDatasetAsset>,
}

impl PublicDatasetLock {
    fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            bail!(
                "unsupported public-dataset lock schema {}",
                self.schema_version
            );
        }
        if self.id.trim().is_empty()
            || self.version.trim().is_empty()
            || self.license.trim().is_empty()
            || self.attribution.trim().is_empty()
            || self.assets.is_empty()
        {
            bail!("dataset identity, rights, attribution, and assets are required");
        }
        if !self.homepage_url.starts_with("https://") {
            bail!("dataset homepage must use HTTPS");
        }
        let mut paths = HashSet::new();
        for asset in &self.assets {
            let path = Path::new(&asset.path);
            if asset.path.trim().is_empty()
                || path.is_absolute()
                || !path
                    .components()
                    .all(|component| matches!(component, Component::Normal(_)))
            {
                bail!(
                    "dataset asset path must stay below the output directory: {}",
                    asset.path
                );
            }
            if !paths.insert(asset.path.as_str()) {
                bail!("duplicate dataset asset path: {}", asset.path);
            }
            if !asset.url.starts_with("https://") {
                bail!("dataset asset URL must use HTTPS: {}", asset.url);
            }
            if asset.size_bytes == 0 || !is_lower_sha256(&asset.sha256) {
                bail!(
                    "dataset asset {} has an invalid size or SHA-256",
                    asset.path
                );
            }
            if let Some(member) = &asset.zip_member {
                let member_path = Path::new(&member.member_path);
                if member.archive_size_bytes == 0
                    || member.member_path.trim().is_empty()
                    || member_path.is_absolute()
                    || !member_path
                        .components()
                        .all(|component| matches!(component, Component::Normal(_)))
                {
                    bail!(
                        "dataset ZIP member must be a safe relative path: {}",
                        member.member_path
                    );
                }
            }
        }
        Ok(())
    }
}

/// Whether a fetched dataset asset was already valid or downloaded now.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DatasetFetchStatus {
    /// The destination already matched the locked size and digest.
    Reused,
    /// New bytes were downloaded and verified before installation.
    Downloaded,
}

/// Verified result for one selected public-dataset artifact.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetFetchAsset {
    /// Relative path copied from the lock.
    pub path: String,
    /// Immutable artifact role.
    pub role: PublicDatasetAssetRole,
    /// Locked and verified SHA-256.
    pub sha256: String,
    /// Reused or downloaded status.
    pub status: DatasetFetchStatus,
}

/// Machine-readable result of fetching one public-dataset slice.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DatasetFetchReport {
    /// Report schema version.
    pub schema_version: u32,
    /// Stable dataset identifier.
    pub dataset_id: String,
    /// Locked upstream version.
    pub dataset_version: String,
    /// Successfully verified selected artifacts.
    pub assets: Vec<DatasetFetchAsset>,
}

/// Fetch and verify a public dataset below an explicit directory.
///
/// Audio is always selected. Annotation-source artifacts are fetched only when
/// `include_annotations` is true. Existing files are trusted only after both
/// their exact byte length and SHA-256 match the lock.
///
/// # Errors
///
/// Returns an error for an invalid lock, unsafe output path, network failure,
/// or content that does not match the immutable identity in the lock.
pub fn fetch_public_dataset(
    lock_path: &Path,
    output_directory: &Path,
    include_annotations: bool,
) -> Result<DatasetFetchReport> {
    let bytes = fs::read(lock_path)
        .with_context(|| format!("reading dataset lock {}", lock_path.display()))?;
    let lock: PublicDatasetLock = serde_json::from_slice(&bytes)
        .with_context(|| format!("parsing dataset lock {}", lock_path.display()))?;
    lock.validate()?;
    fs::create_dir_all(output_directory)
        .with_context(|| format!("creating dataset output {}", output_directory.display()))?;
    let root = output_directory
        .canonicalize()
        .with_context(|| format!("resolving dataset output {}", output_directory.display()))?;
    let agent = Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(180)))
        .user_agent("NagareWorks-rhythm-map-eval/0.1")
        .build()
        .new_agent();
    let mut results = Vec::new();
    let mut zip_directories = HashMap::<(String, u64), Vec<u8>>::new();
    for asset in &lock.assets {
        if asset.role == PublicDatasetAssetRole::AnnotationSource && !include_annotations {
            continue;
        }
        let destination = safe_destination(&root, &asset.path)?;
        let status = if verify_asset(&destination, asset)? {
            DatasetFetchStatus::Reused
        } else {
            eprintln!("fetching {}", asset.path);
            let zip_directory = match &asset.zip_member {
                Some(source) => {
                    let key = (asset.url.clone(), source.archive_size_bytes);
                    if !zip_directories.contains_key(&key) {
                        let directory = fetch_zip_central_directory(
                            &agent,
                            &asset.url,
                            source.archive_size_bytes,
                        )?;
                        zip_directories.insert(key.clone(), directory);
                    }
                    zip_directories.get(&key).map(Vec::as_slice)
                }
                None => None,
            };
            download_verified(&agent, asset, &destination, zip_directory)?;
            DatasetFetchStatus::Downloaded
        };
        results.push(DatasetFetchAsset {
            path: asset.path.clone(),
            role: asset.role,
            sha256: asset.sha256.clone(),
            status,
        });
    }
    Ok(DatasetFetchReport {
        schema_version: 1,
        dataset_id: lock.id,
        dataset_version: lock.version,
        assets: results,
    })
}

/// Fetch selected members from an immutable remote ZIP and identify their
/// extracted bytes for a completed dataset lock.
///
/// This is acquisition tooling, not a relaxed fetch path: destinations and ZIP
/// member names must remain below their declared roots, the server must honor
/// exact byte ranges, and every installed member is hashed after an atomic
/// extraction. Existing destination files are replaced from the named archive
/// rather than trusted before their lock identity exists.
///
/// # Errors
///
/// Returns an error for unsafe paths, duplicate destinations, invalid HTTPS
/// sources, malformed ZIP metadata, failed byte ranges, or extraction errors.
pub fn acquire_public_zip_assets(
    selections: &[PublicDatasetZipAssetSelection],
    output_directory: &Path,
) -> Result<Vec<PublicDatasetAsset>> {
    acquire_public_zip_assets_with_resolution(selections, output_directory, &[])
}

/// Acquire selected ZIP members using optional, TLS-preserving host overrides.
///
/// Host overrides affect connection routing only and are deliberately omitted
/// from the resulting content-addressed lock.
///
/// # Errors
///
/// Returns the same errors as [`acquire_public_zip_assets`], and rejects empty,
/// duplicate, unspecified, multicast, or selection-unrelated host overrides.
pub fn acquire_public_zip_assets_with_resolution(
    selections: &[PublicDatasetZipAssetSelection],
    output_directory: &Path,
    resolutions: &[PublicDatasetHostResolution],
) -> Result<Vec<PublicDatasetAsset>> {
    if selections.is_empty() {
        bail!("ZIP asset acquisition requires at least one selected member");
    }
    let mut paths = HashSet::new();
    for selection in selections {
        validate_zip_asset_selection(selection, &mut paths)?;
    }
    let agent = acquisition_agent(selections, resolutions)?;
    fs::create_dir_all(output_directory)
        .with_context(|| format!("creating dataset output {}", output_directory.display()))?;
    let root = output_directory
        .canonicalize()
        .with_context(|| format!("resolving dataset output {}", output_directory.display()))?;
    let mut zip_directories = HashMap::<(String, u64), Vec<u8>>::new();
    let mut assets = Vec::with_capacity(selections.len());

    for (index, selection) in selections.iter().enumerate() {
        eprintln!(
            "ZIP asset acquisition {}/{}: {}",
            index + 1,
            selections.len(),
            selection.path
        );
        let source = &selection.zip_member;
        let key = (selection.url.clone(), source.archive_size_bytes);
        if !zip_directories.contains_key(&key) {
            let directory =
                fetch_zip_central_directory(&agent, &selection.url, source.archive_size_bytes)?;
            zip_directories.insert(key.clone(), directory);
        }
        let directory = zip_directories
            .get(&key)
            .context("ZIP central directory disappeared during acquisition")?;
        let entry = find_zip_entry(directory, &source.member_path)?;
        if entry.uncompressed_size == 0 {
            bail!("ZIP member {} is empty", source.member_path);
        }
        let destination = safe_destination(&root, &selection.path)?;
        let filename = destination
            .file_name()
            .and_then(|value| value.to_str())
            .context("dataset asset filename is not UTF-8")?;
        let temporary = destination.with_file_name(format!(".{filename}.lock.part"));
        reject_symlink(&temporary)?;
        if temporary.exists() {
            fs::remove_file(&temporary)
                .with_context(|| format!("removing stale acquisition {}", temporary.display()))?;
        }
        let mut asset = PublicDatasetAsset {
            path: selection.path.clone(),
            url: selection.url.clone(),
            sha256: "0".repeat(64),
            size_bytes: entry.uncompressed_size,
            role: selection.role,
            zip_member: Some(source.clone()),
        };
        download_zip_member(&agent, &asset, source, directory, &temporary)
            .with_context(|| format!("acquiring {}", selection.path))?;
        asset.sha256 = sha256_file(&temporary)?;
        replace_regular_file(&temporary, &destination)?;
        assets.push(asset);
    }
    Ok(assets)
}

fn acquisition_agent(
    selections: &[PublicDatasetZipAssetSelection],
    resolutions: &[PublicDatasetHostResolution],
) -> Result<Agent> {
    let config = Agent::config_builder()
        .timeout_global(Some(Duration::from_secs(180)))
        .user_agent("NagareWorks-rhythm-map-eval/0.1")
        .build();
    if resolutions.is_empty() {
        return Ok(config.new_agent());
    }

    let source_hosts = selections
        .iter()
        .map(|selection| {
            selection
                .url
                .parse::<Uri>()
                .map_err(anyhow::Error::from)
                .and_then(|uri| {
                    uri.host()
                        .map(str::to_ascii_lowercase)
                        .context("dataset ZIP URL has no host")
                })
        })
        .collect::<Result<HashSet<_>>>()?;
    let mut overrides = HashMap::<String, Vec<IpAddr>>::new();
    for resolution in resolutions {
        let host = resolution.host.trim().to_ascii_lowercase();
        if host.is_empty()
            || host.contains('/')
            || host.contains(':')
            || !source_hosts.contains(&host)
        {
            bail!(
                "host resolution override must name a selected ZIP host: {}",
                resolution.host
            );
        }
        if resolution.address.is_unspecified() || resolution.address.is_multicast() {
            bail!(
                "host resolution override has an unusable address: {}",
                resolution.address
            );
        }
        let addresses = overrides.entry(host).or_default();
        if addresses.contains(&resolution.address) {
            bail!(
                "duplicate host resolution override: {}={}",
                resolution.host,
                resolution.address
            );
        }
        addresses.push(resolution.address);
    }
    Ok(Agent::with_parts(
        config,
        DefaultConnector::default(),
        AcquisitionResolver {
            overrides,
            fallback: DefaultResolver::default(),
        },
    ))
}

fn validate_zip_asset_selection<'a>(
    selection: &'a PublicDatasetZipAssetSelection,
    paths: &mut HashSet<&'a str>,
) -> Result<()> {
    let path = Path::new(&selection.path);
    if selection.path.trim().is_empty()
        || path.is_absolute()
        || !path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        bail!(
            "dataset asset path must stay below the output directory: {}",
            selection.path
        );
    }
    if !paths.insert(&selection.path) {
        bail!("duplicate dataset asset path: {}", selection.path);
    }
    if !selection.url.starts_with("https://") {
        bail!("dataset asset URL must use HTTPS: {}", selection.url);
    }
    let source = &selection.zip_member;
    let member_path = Path::new(&source.member_path);
    if source.archive_size_bytes == 0
        || source.member_path.trim().is_empty()
        || member_path.is_absolute()
        || !member_path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
    {
        bail!(
            "dataset ZIP member must be a safe relative path: {}",
            source.member_path
        );
    }
    Ok(())
}

fn safe_destination(root: &Path, relative: &str) -> Result<PathBuf> {
    let relative = Path::new(relative);
    let parent = relative
        .parent()
        .context("dataset asset destination has no parent")?;
    let mut resolved_parent = root.to_path_buf();
    for component in parent.components() {
        let Component::Normal(name) = component else {
            bail!("dataset asset destination escapes the output directory");
        };
        let candidate = resolved_parent.join(name);
        match fs::symlink_metadata(&candidate) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                bail!(
                    "dataset directory may not be a symbolic link: {}",
                    candidate.display()
                );
            }
            Ok(metadata) if !metadata.is_dir() => {
                bail!(
                    "dataset directory path is not a directory: {}",
                    candidate.display()
                );
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                fs::create_dir(&candidate).with_context(|| {
                    format!("creating dataset directory {}", candidate.display())
                })?;
            }
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("reading dataset directory {}", candidate.display()));
            }
        }
        resolved_parent = candidate
            .canonicalize()
            .with_context(|| format!("resolving dataset directory {}", candidate.display()))?;
        if !resolved_parent.starts_with(root) {
            bail!("dataset asset destination escapes the output directory");
        }
    }
    Ok(resolved_parent.join(
        relative
            .file_name()
            .context("dataset asset destination has no filename")?,
    ))
}

fn verify_asset(path: &Path, asset: &PublicDatasetAsset) -> Result<bool> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error).with_context(|| format!("reading {}", path.display())),
    };
    if metadata.file_type().is_symlink() {
        bail!(
            "dataset destination may not be a symbolic link: {}",
            path.display()
        );
    }
    if !metadata.is_file() || metadata.len() != asset.size_bytes {
        return Ok(false);
    }
    Ok(sha256_file(path)? == asset.sha256)
}

fn download_verified(
    agent: &Agent,
    asset: &PublicDatasetAsset,
    destination: &Path,
    zip_directory: Option<&[u8]>,
) -> Result<()> {
    let filename = destination
        .file_name()
        .and_then(|value| value.to_str())
        .context("dataset asset filename is not UTF-8")?;
    let temporary = destination.with_file_name(format!(".{filename}.part"));
    reject_symlink(&temporary)?;
    let mut last_error = None;
    for attempt in 1..=DOWNLOAD_ATTEMPTS {
        if temporary.exists() {
            fs::remove_file(&temporary)
                .with_context(|| format!("removing stale download {}", temporary.display()))?;
        }
        let result = download_once(agent, asset, &temporary, zip_directory).and_then(|()| {
            if verify_asset(&temporary, asset)? {
                Ok(())
            } else {
                bail!("downloaded bytes did not match locked size and SHA-256")
            }
        });
        match result {
            Ok(()) => {
                replace_regular_file(&temporary, destination)?;
                return Ok(());
            }
            Err(error) => {
                last_error = Some(error.context(format!(
                    "attempt {attempt}/{DOWNLOAD_ATTEMPTS} for {}",
                    asset.path
                )));
            }
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow!("dataset download failed")))
}

fn download_once(
    agent: &Agent,
    asset: &PublicDatasetAsset,
    temporary: &Path,
    zip_directory: Option<&[u8]>,
) -> Result<()> {
    if let Some(member) = &asset.zip_member {
        return download_zip_member(
            agent,
            asset,
            member,
            zip_directory.context("ZIP member download is missing its central directory")?,
            temporary,
        );
    }
    let mut response = agent
        .get(&asset.url)
        .call()
        .with_context(|| format!("requesting {}", asset.url))?;
    let mut file = File::create(temporary)
        .with_context(|| format!("creating download {}", temporary.display()))?;
    io::copy(&mut response.body_mut().as_reader(), &mut file)
        .with_context(|| format!("downloading {}", asset.url))?;
    file.sync_all()
        .with_context(|| format!("syncing download {}", temporary.display()))?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ZipDirectory {
    offset: u64,
    size: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ZipEntry {
    compression_method: u16,
    compressed_size: u64,
    uncompressed_size: u64,
    local_header_offset: u64,
}

#[allow(clippy::too_many_lines)]
fn download_zip_member(
    agent: &Agent,
    asset: &PublicDatasetAsset,
    source: &PublicDatasetZipMember,
    central_directory: &[u8],
    temporary: &Path,
) -> Result<()> {
    let entry = find_zip_entry(central_directory, &source.member_path)?;
    if entry.uncompressed_size != asset.size_bytes {
        bail!(
            "ZIP member {} has {} bytes, lock expects {}",
            source.member_path,
            entry.uncompressed_size,
            asset.size_bytes
        );
    }
    if entry.compressed_size == 0 {
        bail!("ZIP member {} is empty", source.member_path);
    }

    // A local ZIP extra field is at most u16::MAX bytes. Fetching that bounded
    // header allowance avoids separate name and extra-field round trips while
    // keeping the much larger compressed payload available for chunked ranges.
    let maximum_header_size = 30_u64
        .checked_add(u64::try_from(source.member_path.len())?)
        .and_then(|size| size.checked_add(u64::from(u16::MAX)))
        .context("ZIP member header request size overflow")?;
    let header_end = checked_range_end(
        entry.local_header_offset,
        maximum_header_size,
        "ZIP member header request",
    )?
    .min(source.archive_size_bytes - 1);
    let header =
        request_range_with_retries(agent, &asset.url, entry.local_header_offset, header_end)?;
    if read_u32(&header, 0)? != 0x0403_4b50 {
        bail!(
            "ZIP member {} has an invalid local header",
            source.member_path
        );
    }
    let name_length = usize::from(read_u16(&header, 26)?);
    let extra_length = usize::from(read_u16(&header, 28)?);
    let name_start = 30_usize;
    let name_end = name_start
        .checked_add(name_length)
        .context("ZIP local member name overflow")?;
    let data_start = name_end
        .checked_add(extra_length)
        .context("ZIP local member header overflow")?;
    let local_name = std::str::from_utf8(
        header
            .get(name_start..name_end)
            .context("ZIP local member name is truncated")?,
    )
    .context("ZIP local member name is not UTF-8")?;
    if local_name != source.member_path {
        bail!(
            "ZIP local member name {local_name:?} does not match {:?}",
            source.member_path
        );
    }
    if header.get(name_end..data_start).is_none() {
        bail!("ZIP local member extra field is truncated");
    }
    let absolute_data_start = entry
        .local_header_offset
        .checked_add(u64::try_from(data_start)?)
        .context("ZIP member data start overflow")?;
    let absolute_data_end = absolute_data_start
        .checked_add(entry.compressed_size)
        .context("ZIP member data end overflow")?;
    if absolute_data_end > source.archive_size_bytes {
        bail!(
            "ZIP member {} exceeds the locked archive",
            source.member_path
        );
    }
    let compressed = request_range_chunked(
        agent,
        &asset.url,
        absolute_data_start,
        entry.compressed_size,
    )?;
    let mut output = File::create(temporary)
        .with_context(|| format!("creating extracted member {}", temporary.display()))?;
    let output_limit = asset
        .size_bytes
        .checked_add(1)
        .context("ZIP member output limit overflow")?;
    let written = match entry.compression_method {
        0 => {
            if entry.compressed_size != entry.uncompressed_size {
                bail!(
                    "stored ZIP member {} has inconsistent sizes",
                    source.member_path
                );
            }
            io::copy(
                &mut Cursor::new(&compressed).take(output_limit),
                &mut output,
            )
        }
        8 => io::copy(
            &mut DeflateDecoder::new(Cursor::new(&compressed)).take(output_limit),
            &mut output,
        ),
        method => bail!(
            "ZIP member {} uses unsupported compression method {method}",
            source.member_path
        ),
    }
    .with_context(|| format!("extracting ZIP member {}", source.member_path))?;
    if written != asset.size_bytes {
        bail!(
            "ZIP member {} expanded to {written} bytes, lock expects {}",
            source.member_path,
            asset.size_bytes
        );
    }
    output
        .sync_all()
        .with_context(|| format!("syncing extracted member {}", temporary.display()))?;
    Ok(())
}

fn request_range_chunked(agent: &Agent, url: &str, start: u64, size: u64) -> Result<Vec<u8>> {
    let end = checked_range_end(start, size, "chunked HTTP range")?;
    let mut ranges = Vec::new();
    let mut chunk_start = start;
    while chunk_start <= end {
        let chunk_end = chunk_start
            .saturating_add(ZIP_RANGE_CHUNK_BYTES - 1)
            .min(end);
        ranges.push((chunk_start, chunk_end));
        chunk_start = chunk_end
            .checked_add(1)
            .context("chunked HTTP range overflow")?;
    }

    let mut bytes = Vec::with_capacity(usize::try_from(size)?);
    for batch in ranges.chunks(ZIP_MEMBER_RANGE_CONCURRENCY) {
        let results = std::thread::scope(|scope| {
            batch
                .iter()
                .map(|&(chunk_start, chunk_end)| {
                    scope.spawn(move || {
                        request_range_with_retries(agent, url, chunk_start, chunk_end)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .map(|handle| {
                    handle
                        .join()
                        .map_err(|_| anyhow!("ZIP member range worker panicked"))?
                })
                .collect::<Result<Vec<_>>>()
        })?;
        for chunk in results {
            bytes.extend_from_slice(&chunk);
        }
    }
    if u64::try_from(bytes.len())? != size {
        bail!(
            "chunked HTTP range returned {} bytes, expected {size}",
            bytes.len()
        );
    }
    Ok(bytes)
}

fn fetch_zip_central_directory(agent: &Agent, url: &str, archive_size: u64) -> Result<Vec<u8>> {
    let directory = fetch_zip_directory(agent, url, archive_size)?;
    if directory.size > MAX_CENTRAL_DIRECTORY_BYTES {
        bail!(
            "ZIP central directory is too large to inspect safely: {} bytes",
            directory.size
        );
    }
    request_range_chunked(agent, url, directory.offset, directory.size)
}

fn fetch_zip_directory(agent: &Agent, url: &str, archive_size: u64) -> Result<ZipDirectory> {
    if archive_size < 22 {
        bail!("ZIP archive is smaller than its end record");
    }
    let tail_size = archive_size.min(ZIP_TAIL_BYTES);
    let tail_offset = archive_size - tail_size;
    let tail = request_range_chunked(agent, url, tail_offset, tail_size)?;
    let eocd = tail
        .windows(4)
        .rposition(|bytes| bytes == [0x50, 0x4b, 0x05, 0x06])
        .context("ZIP end-of-central-directory record was not found")?;
    let directory_size = u64::from(read_u32(&tail, eocd + 12)?);
    let directory_offset = u64::from(read_u32(&tail, eocd + 16)?);
    if directory_size != u64::from(u32::MAX) && directory_offset != u64::from(u32::MAX) {
        return Ok(ZipDirectory {
            offset: directory_offset,
            size: directory_size,
        });
    }

    let locator = eocd
        .checked_sub(20)
        .context("ZIP64 end locator is missing")?;
    if read_u32(&tail, locator)? != 0x0706_4b50 {
        bail!("ZIP64 end locator has an invalid signature");
    }
    let zip64_offset = read_u64(&tail, locator + 8)?;
    let zip64_end = checked_range_end(zip64_offset, 56, "ZIP64 end record")?;
    let zip64 = request_range_with_retries(agent, url, zip64_offset, zip64_end)?;
    if read_u32(&zip64, 0)? != 0x0606_4b50 {
        bail!("ZIP64 end record has an invalid signature");
    }
    Ok(ZipDirectory {
        size: read_u64(&zip64, 40)?,
        offset: read_u64(&zip64, 48)?,
    })
}

fn find_zip_entry(directory: &[u8], member_path: &str) -> Result<ZipEntry> {
    let mut cursor = 0_usize;
    while cursor < directory.len() {
        if read_u32(directory, cursor)? != 0x0201_4b50 {
            bail!("ZIP central directory has an invalid entry at byte {cursor}");
        }
        let flags = read_u16(directory, cursor + 8)?;
        let compression_method = read_u16(directory, cursor + 10)?;
        let compressed_size_32 = read_u32(directory, cursor + 20)?;
        let uncompressed_size_32 = read_u32(directory, cursor + 24)?;
        let name_length = usize::from(read_u16(directory, cursor + 28)?);
        let extra_length = usize::from(read_u16(directory, cursor + 30)?);
        let comment_length = usize::from(read_u16(directory, cursor + 32)?);
        let local_offset_32 = read_u32(directory, cursor + 42)?;
        let name_start = cursor
            .checked_add(46)
            .context("ZIP central entry offset overflow")?;
        let name_end = name_start
            .checked_add(name_length)
            .context("ZIP central entry name overflow")?;
        let extra_end = name_end
            .checked_add(extra_length)
            .context("ZIP central entry extra data overflow")?;
        let entry_end = extra_end
            .checked_add(comment_length)
            .context("ZIP central entry comment overflow")?;
        let name = std::str::from_utf8(
            directory
                .get(name_start..name_end)
                .context("ZIP central entry name is truncated")?,
        )
        .context("ZIP central member name is not UTF-8")?;
        let extra = directory
            .get(name_end..extra_end)
            .context("ZIP central entry extra data is truncated")?;
        if name == member_path {
            if flags & 1 != 0 {
                bail!("ZIP member {member_path} is encrypted");
            }
            let (uncompressed_size, compressed_size, local_header_offset) = resolve_zip64_sizes(
                extra,
                uncompressed_size_32,
                compressed_size_32,
                local_offset_32,
            )?;
            return Ok(ZipEntry {
                compression_method,
                compressed_size,
                uncompressed_size,
                local_header_offset,
            });
        }
        cursor = entry_end;
    }
    bail!("ZIP member not found: {member_path}")
}

fn resolve_zip64_sizes(
    extra: &[u8],
    uncompressed_size_32: u32,
    compressed_size_32: u32,
    local_offset_32: u32,
) -> Result<(u64, u64, u64)> {
    let needs_zip64 = uncompressed_size_32 == u32::MAX
        || compressed_size_32 == u32::MAX
        || local_offset_32 == u32::MAX;
    if !needs_zip64 {
        return Ok((
            u64::from(uncompressed_size_32),
            u64::from(compressed_size_32),
            u64::from(local_offset_32),
        ));
    }
    let mut cursor = 0_usize;
    while cursor < extra.len() {
        let id = read_u16(extra, cursor)?;
        let size = usize::from(read_u16(extra, cursor + 2)?);
        let data_start = cursor
            .checked_add(4)
            .context("ZIP extra field offset overflow")?;
        let data_end = data_start
            .checked_add(size)
            .context("ZIP extra field size overflow")?;
        let data = extra
            .get(data_start..data_end)
            .context("ZIP extra field is truncated")?;
        if id == 1 {
            let mut zip64_cursor = 0_usize;
            let uncompressed_size = if uncompressed_size_32 == u32::MAX {
                let value = read_u64(data, zip64_cursor)?;
                zip64_cursor += 8;
                value
            } else {
                u64::from(uncompressed_size_32)
            };
            let compressed_size = if compressed_size_32 == u32::MAX {
                let value = read_u64(data, zip64_cursor)?;
                zip64_cursor += 8;
                value
            } else {
                u64::from(compressed_size_32)
            };
            let local_header_offset = if local_offset_32 == u32::MAX {
                read_u64(data, zip64_cursor)?
            } else {
                u64::from(local_offset_32)
            };
            return Ok((uncompressed_size, compressed_size, local_header_offset));
        }
        cursor = data_end;
    }
    bail!("ZIP64 entry is missing its ZIP64 extra field")
}

fn request_range(agent: &Agent, url: &str, start: u64, end: u64) -> Result<Vec<u8>> {
    if end < start {
        bail!("invalid HTTP byte range {start}-{end}");
    }
    let expected = end - start + 1;
    let capacity = usize::try_from(expected).context("HTTP byte range is too large")?;
    let mut response = agent
        .get(url)
        .header("Range", format!("bytes={start}-{end}"))
        .call()
        .with_context(|| format!("requesting bytes {start}-{end} from {url}"))?;
    if response.status().as_u16() != 206 {
        bail!(
            "server returned HTTP {} instead of 206 for byte range {start}-{end}",
            response.status().as_u16()
        );
    }
    let mut bytes = Vec::with_capacity(capacity);
    response
        .body_mut()
        .as_reader()
        .take(expected + 1)
        .read_to_end(&mut bytes)
        .with_context(|| format!("downloading bytes {start}-{end} from {url}"))?;
    if bytes.len() != capacity {
        bail!(
            "server returned {} bytes for range {start}-{end}, expected {expected}",
            bytes.len()
        );
    }
    Ok(bytes)
}

fn request_range_with_retries(agent: &Agent, url: &str, start: u64, end: u64) -> Result<Vec<u8>> {
    let mut last_error = None;
    for attempt in 1..=DOWNLOAD_ATTEMPTS {
        match request_range(agent, url, start, end) {
            Ok(bytes) => return Ok(bytes),
            Err(error) => {
                last_error = Some(error.context(format!(
                    "range attempt {attempt}/{DOWNLOAD_ATTEMPTS} for bytes {start}-{end}"
                )));
            }
        }
    }
    Err(last_error.unwrap_or_else(|| anyhow!("HTTP range request failed")))
}

fn checked_range_end(start: u64, size: u64, label: &str) -> Result<u64> {
    if size == 0 {
        bail!("{label} is empty");
    }
    start
        .checked_add(size - 1)
        .with_context(|| format!("{label} byte range overflow"))
}

fn read_u16(bytes: &[u8], offset: usize) -> Result<u16> {
    let value = bytes
        .get(offset..offset + 2)
        .context("ZIP structure is truncated")?;
    Ok(u16::from_le_bytes([value[0], value[1]]))
}

fn read_u32(bytes: &[u8], offset: usize) -> Result<u32> {
    let value = bytes
        .get(offset..offset + 4)
        .context("ZIP structure is truncated")?;
    Ok(u32::from_le_bytes([value[0], value[1], value[2], value[3]]))
}

fn read_u64(bytes: &[u8], offset: usize) -> Result<u64> {
    let value = bytes
        .get(offset..offset + 8)
        .context("ZIP structure is truncated")?;
    Ok(u64::from_le_bytes([
        value[0], value[1], value[2], value[3], value[4], value[5], value[6], value[7],
    ]))
}

fn replace_regular_file(temporary: &Path, destination: &Path) -> Result<()> {
    reject_symlink(destination)?;
    if destination.exists() {
        fs::remove_file(destination)
            .with_context(|| format!("replacing invalid asset {}", destination.display()))?;
    }
    fs::rename(temporary, destination).with_context(|| {
        format!(
            "installing verified asset {} as {}",
            temporary.display(),
            destination.display()
        )
    })
}

fn reject_symlink(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(metadata) if metadata.file_type().is_symlink() => {
            bail!(
                "dataset path may not be a symbolic link: {}",
                path.display()
            )
        }
        Ok(_) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
    }
}

fn is_lower_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU64, Ordering};

    use sha2::{Digest, Sha256};

    use super::*;

    static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

    fn temporary_directory() -> PathBuf {
        let suffix = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        std::env::temp_dir().join(format!(
            "rhythm-map-public-dataset-{}-{suffix}",
            std::process::id()
        ))
    }

    fn asset(bytes: &[u8]) -> PublicDatasetAsset {
        PublicDatasetAsset {
            path: "audio/example.mp3".to_string(),
            url: "https://example.invalid/example.mp3".to_string(),
            sha256: format!("{:x}", Sha256::digest(bytes)),
            size_bytes: bytes.len() as u64,
            role: PublicDatasetAssetRole::Audio,
            zip_member: None,
        }
    }

    #[test]
    fn zip_acquisition_requires_a_selected_member() {
        let error = acquire_public_zip_assets(&[], Path::new("unused"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("at least one selected member"));
    }

    #[test]
    fn zip_asset_selection_rejects_an_escaping_destination_before_network() {
        let selection = PublicDatasetZipAssetSelection {
            path: "../audio.wav".to_string(),
            url: "https://example.invalid/archive.zip".to_string(),
            role: PublicDatasetAssetRole::Audio,
            zip_member: PublicDatasetZipMember {
                archive_size_bytes: 1_024,
                member_path: "audio.wav".to_string(),
            },
        };
        let root = temporary_directory();
        let error = acquire_public_zip_assets(&[selection], &root)
            .unwrap_err()
            .to_string();
        assert!(error.contains("below the output directory"));
        assert!(!root.exists());
    }

    #[test]
    fn zip_acquisition_rejects_unrelated_host_override_before_network() {
        let selection = PublicDatasetZipAssetSelection {
            path: "audio.wav".to_string(),
            url: "https://example.invalid/archive.zip".to_string(),
            role: PublicDatasetAssetRole::Audio,
            zip_member: PublicDatasetZipMember {
                archive_size_bytes: 1_024,
                member_path: "audio.wav".to_string(),
            },
        };
        let resolution = PublicDatasetHostResolution {
            host: "other.invalid".to_string(),
            address: "127.0.0.1".parse().unwrap(),
        };
        let root = temporary_directory();
        let error = acquire_public_zip_assets_with_resolution(&[selection], &root, &[resolution])
            .unwrap_err()
            .to_string();
        assert!(error.contains("selected ZIP host"));
        assert!(!root.exists());
    }

    #[test]
    fn existing_asset_requires_size_and_digest() {
        let root = temporary_directory();
        fs::create_dir_all(root.join("audio")).unwrap();
        let expected = asset(b"expected bytes");
        let path = root.join(&expected.path);
        fs::write(&path, b"same byte count!").unwrap();
        assert!(!verify_asset(&path, &expected).unwrap());
        fs::write(&path, b"expected bytes").unwrap();
        assert!(verify_asset(&path, &expected).unwrap());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn valid_existing_asset_is_reused_without_network() {
        let root = temporary_directory();
        let output = root.join("output");
        fs::create_dir_all(output.join("audio")).unwrap();
        let expected = asset(b"expected bytes");
        fs::write(output.join(&expected.path), b"expected bytes").unwrap();
        let lock = PublicDatasetLock {
            schema_version: 1,
            id: "fixture".to_string(),
            version: "1".to_string(),
            homepage_url: "https://example.invalid".to_string(),
            license: "CC-BY-4.0".to_string(),
            attribution: "Fixture author".to_string(),
            assets: vec![expected],
        };
        let lock_path = root.join("lock.json");
        fs::write(&lock_path, serde_json::to_vec(&lock).unwrap()).unwrap();
        let report = fetch_public_dataset(&lock_path, &output, false).unwrap();
        assert_eq!(report.assets[0].status, DatasetFetchStatus::Reused);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn lock_rejects_paths_outside_the_output_root() {
        let mut escaped = asset(b"expected bytes");
        escaped.path = "../example.mp3".to_string();
        let lock = PublicDatasetLock {
            schema_version: 1,
            id: "fixture".to_string(),
            version: "1".to_string(),
            homepage_url: "https://example.invalid".to_string(),
            license: "CC-BY-4.0".to_string(),
            attribution: "Fixture author".to_string(),
            assets: vec![escaped],
        };

        assert!(lock.validate().is_err());
    }

    #[test]
    fn lock_rejects_unsafe_zip_member_paths() {
        let mut archived = asset(b"expected bytes");
        archived.zip_member = Some(PublicDatasetZipMember {
            archive_size_bytes: 1_024,
            member_path: "../audio/example.mp3".to_string(),
        });
        let lock = PublicDatasetLock {
            schema_version: 1,
            id: "fixture".to_string(),
            version: "1".to_string(),
            homepage_url: "https://example.invalid".to_string(),
            license: "CC-BY-4.0".to_string(),
            attribution: "Fixture author".to_string(),
            assets: vec![archived],
        };

        assert!(lock.validate().is_err());
    }

    #[test]
    fn pinned_vienna_dataset_lock_is_valid() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../evaluation/datasets/vienna4x22-holdout-v1.json");
        let lock: PublicDatasetLock =
            serde_json::from_slice(&fs::read(path).expect("read Vienna dataset lock"))
                .expect("parse Vienna dataset lock");
        lock.validate().expect("validate Vienna dataset lock");
        assert_eq!(lock.assets.len(), 12);
        assert!(lock.assets.iter().all(|asset| asset.zip_member.is_some()));
    }

    #[test]
    fn pinned_rubato_dataset_lock_is_valid() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../evaluation/datasets/rubato-calibration-v1.json");
        let lock: PublicDatasetLock =
            serde_json::from_slice(&fs::read(path).expect("read RUBATO dataset lock"))
                .expect("parse RUBATO dataset lock");
        lock.validate().expect("validate RUBATO dataset lock");
        assert_eq!(lock.assets.len(), 102);
        assert_eq!(
            lock.assets
                .iter()
                .filter(|asset| asset.role == PublicDatasetAssetRole::Audio)
                .count(),
            25
        );
        assert_eq!(
            lock.assets
                .iter()
                .filter(|asset| asset.zip_member.is_some())
                .count(),
            100
        );
    }

    #[test]
    fn pinned_rubato_holdout_lock_is_valid() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../evaluation/datasets/rubato-holdout-v1.json");
        let lock: PublicDatasetLock =
            serde_json::from_slice(&fs::read(path).expect("read RUBATO holdout lock"))
                .expect("parse RUBATO holdout lock");
        lock.validate().expect("validate RUBATO holdout lock");
        assert_eq!(lock.id, "rubato-holdout-v1");
        assert_eq!(lock.assets.len(), 18);
        assert_eq!(
            lock.assets
                .iter()
                .filter(|asset| asset.role == PublicDatasetAssetRole::Audio)
                .count(),
            4
        );
        assert_eq!(
            lock.assets
                .iter()
                .filter(|asset| asset.zip_member.is_some())
                .count(),
            16
        );
    }

    #[test]
    fn central_directory_resolves_regular_and_zip64_entries() {
        let regular = central_entry("audio/regular.wav", 80, 100, 42, &[]);
        assert_eq!(
            find_zip_entry(&regular, "audio/regular.wav").unwrap(),
            ZipEntry {
                compression_method: 8,
                compressed_size: 80,
                uncompressed_size: 100,
                local_header_offset: 42,
            }
        );

        let mut zip64_extra = Vec::new();
        push_u16(&mut zip64_extra, 1);
        push_u16(&mut zip64_extra, 24);
        push_u64(&mut zip64_extra, 5_000_000_100);
        push_u64(&mut zip64_extra, 5_000_000_080);
        push_u64(&mut zip64_extra, 4_900_000_000);
        let zip64 = central_entry(
            "audio/zip64.wav",
            u32::MAX,
            u32::MAX,
            u32::MAX,
            &zip64_extra,
        );
        assert_eq!(
            find_zip_entry(&zip64, "audio/zip64.wav").unwrap(),
            ZipEntry {
                compression_method: 8,
                compressed_size: 5_000_000_080,
                uncompressed_size: 5_000_000_100,
                local_header_offset: 4_900_000_000,
            }
        );
    }

    fn central_entry(
        name: &str,
        compressed_size: u32,
        uncompressed_size: u32,
        local_header_offset: u32,
        extra: &[u8],
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        push_u32(&mut bytes, 0x0201_4b50);
        push_u16(&mut bytes, 45);
        push_u16(&mut bytes, 45);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 8);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 0);
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, compressed_size);
        push_u32(&mut bytes, uncompressed_size);
        push_u16(&mut bytes, u16::try_from(name.len()).unwrap());
        push_u16(&mut bytes, u16::try_from(extra.len()).unwrap());
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 0);
        push_u16(&mut bytes, 0);
        push_u32(&mut bytes, 0);
        push_u32(&mut bytes, local_header_offset);
        bytes.extend_from_slice(name.as_bytes());
        bytes.extend_from_slice(extra);
        bytes
    }

    fn push_u16(bytes: &mut Vec<u8>, value: u16) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u32(bytes: &mut Vec<u8>, value: u32) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_u64(bytes: &mut Vec<u8>, value: u64) {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
}
