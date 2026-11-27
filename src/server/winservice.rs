//! Running as a Windows service, so that the service works whoever is (or is
//! not) signed in. `scripts/install-windows-service.ps1` sets it up.

use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result};
use windows_service::service::{
    ServiceControl, ServiceControlAccept, ServiceExitCode, ServiceState, ServiceStatus, ServiceType,
};
use windows_service::service_control_handler::{self, ServiceControlHandlerResult};
use windows_service::{define_windows_service, service_dispatcher};

use super::Config;

pub const NAME: &str = "ChitraEnhancer";

static CONFIG: OnceLock<PathBuf> = OnceLock::new();

define_windows_service!(entry, service_main);

/// Hands the process to the Windows service manager. Fails when the program was
/// not started by it.
pub fn run(config: PathBuf) -> Result<()> {
    let _ = CONFIG.set(config);
    service_dispatcher::start(NAME, entry)
        .context("--windows-service only works when Windows starts the program as a service")
}

fn service_main(_: Vec<OsString>) {
    // There is no console: faults are noted in the log by `serve`.
    let _ = serve();
}

fn serve() -> Result<()> {
    let stop = Arc::new(AtomicBool::new(false));
    let asked = stop.clone();
    let handle = service_control_handler::register(NAME, move |control| match control {
        ServiceControl::Stop | ServiceControl::Shutdown => {
            asked.store(true, Ordering::Relaxed);
            ServiceControlHandlerResult::NoError
        }
        ServiceControl::Interrogate => ServiceControlHandlerResult::NoError,
        _ => ServiceControlHandlerResult::NotImplemented,
    })?;
    let report = |state, code| {
        handle.set_service_status(ServiceStatus {
            service_type: ServiceType::OWN_PROCESS,
            current_state: state,
            controls_accepted: if state == ServiceState::Running {
                ServiceControlAccept::STOP | ServiceControlAccept::SHUTDOWN
            } else {
                ServiceControlAccept::empty()
            },
            exit_code: code,
            checkpoint: 0,
            wait_hint: Duration::from_secs(10),
            process_id: None,
        })
    };
    // Running at once: waiting for Tailscale must not look like a hung start.
    report(ServiceState::Running, ServiceExitCode::Win32(0))?;
    let ran = CONFIG
        .get()
        .context("no configuration file was given")
        .and_then(|file| Config::read(file))
        .and_then(|config| {
            let log = config.clone();
            super::run(config, stop, |_| {})
                .inspect_err(|e| log.log(&format!("stopped by a fault: {e:#}")))
        });
    // A non-zero code makes Windows restart it, as the installer asks.
    let code = if ran.is_ok() {
        ServiceExitCode::Win32(0)
    } else {
        ServiceExitCode::ServiceSpecific(1)
    };
    report(ServiceState::Stopped, code)?;
    ran
}
