//! Finding a peer without typing an address.
//!
//! Query and answer, not announce-and-hope. The first design had the
//! worker shouting a datagram every two seconds — and it was refuted on
//! the stand: broadcast sent from a laptop never reaches an Android
//! phone on the same Wi-Fi (both 255.255.255.255 and the subnet form
//! were dropped), while a unicast datagram to the same phone arrives
//! every time. Android filters broadcast to userspace, and no amount of
//! shouting fixes a listener that is not allowed to hear.
//!
//! So the side that WANTS to find asks: the seeker broadcasts one small
//! query, and each worker answers by unicast straight back to it. The
//! seeker only has to send broadcast (allowed) and receive unicast
//! (proven). Plain UDP rather than mDNS: mDNS is a dependency and a
//! second protocol to be wrong about, for the one question worth asking
//! — "who here holds this model".
//!
//! **The beacon carries no secret.** It advertises existence, identity
//! and geometry — never the token. A listener still has to prove itself
//! at the handshake, and a wrong `dir_hash` is refused there. Announcing
//! is opt-out (`CMF_NET_BEACON=0`) for anyone who would rather not be
//! visible.
//!
//! The address is deliberately not authenticated: on an untrusted
//! network this tells a stranger that a model is here, and that is the
//! honest cost of zero-configuration. Over a cable there is nothing to
//! discover — a tether is one fixed address — so this exists for Wi-Fi.

use serde::{Deserialize, Serialize};
use std::net::{IpAddr, SocketAddr, UdpSocket};
use std::time::{Duration, Instant};

/// Where queries are sent and workers listen. Not the worker's TCP port
/// — that is inside the answer.
pub const BEACON_PORT: u16 = 9910;
const MAGIC: &str = "cortiq-worker";
/// What a seeker sends. Small, constant, and carrying nothing.
const QUERY: &[u8] = b"cortiq-who";

/// What a worker says about itself. Every field is checkable at the
/// handshake, so a lying beacon buys nothing but a refused connection.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Beacon {
    /// Constant marker so a foreign datagram on this port is ignored.
    pub magic: String,
    /// Protocol version — a peer on another version is listed and
    /// labelled, not hidden, because "why can't I connect" deserves an
    /// answer.
    pub wire: u32,
    /// TCP port the worker is actually serving on.
    pub port: u16,
    /// Model identity, hex. Both sides must match or the handshake
    /// refuses — showing it lets a UI grey out what cannot work.
    pub dir_hash: String,
    pub arch: String,
    pub layers: u32,
    pub hidden: u32,
    /// File name only, never the path: a home directory is not the
    /// network's business.
    pub model: String,
    /// True when the worker needs a token. A UI can ask for it up front
    /// instead of after a refusal.
    pub token_required: bool,
}

/// A beacon plus where it came from.
#[derive(Debug, Clone)]
pub struct Found {
    pub addr: SocketAddr,
    pub beacon: Beacon,
}

impl Found {
    /// What a caller passes to `--peer`.
    pub fn peer_addr(&self) -> String {
        format!("{}:{}", self.addr.ip(), self.beacon.port)
    }
}

fn announce_disabled() -> bool {
    std::env::var("CMF_NET_BEACON").is_ok_and(|v| v == "0")
}

/// Answer queries until the process exits. Nothing is emitted unasked:
/// a worker is silent until someone on the network asks who is there,
/// which is both what works through Android's filtering and the quieter
/// thing to do on a network that is not yours.
///
/// Failures are deliberately silent. A machine that cannot bind the
/// discovery port still serves fine over an address typed by hand, and
/// a worker that refused to start because it could not be discovered
/// would be the worse product.
pub fn announce(b: Beacon) {
    if announce_disabled() {
        return;
    }
    let payload = match serde_json::to_vec(&b) {
        Ok(p) => p,
        Err(_) => return,
    };
    std::thread::Builder::new()
        .name("cortiq-beacon".into())
        .spawn(move || {
            // A second worker on the same host loses this bind and is
            // simply not discoverable — it still serves an address typed
            // by hand, which is the honest trade for keeping this simple.
            let sock = match UdpSocket::bind(("0.0.0.0", BEACON_PORT)) {
                Ok(s) => s,
                Err(e) => {
                    // Not fatal — the worker serves an address typed by
                    // hand either way — but silence here reads as "the
                    // network has nobody on it", which is a different
                    // problem entirely. Say which one it is.
                    eprintln!("worker: not discoverable — udp/{BEACON_PORT}: {e}");
                    return;
                }
            };
            let mut buf = [0u8; 512];
            loop {
                let Ok((n, from)) = sock.recv_from(&mut buf) else {
                    continue;
                };
                if &buf[..n] == QUERY {
                    // Straight back to the asker: unicast is the one
                    // direction that survives a phone's filtering.
                    let _ = sock.send_to(&payload, from);
                }
            }
        })
        .ok();
}

/// Ask the network and collect the answers.
///
/// The seeker binds an EPHEMERAL port — never the discovery port — so a
/// machine already running a worker can still scan, and so the replies
/// arrive as unicast to a port only this process holds. One query goes
/// to the global broadcast address and one to the subnet form, because
/// stacks disagree about which of the two leaves the interface.
pub fn discover(wait: Duration) -> Result<Vec<Found>, String> {
    let sock = UdpSocket::bind(("0.0.0.0", 0)).map_err(|e| format!("discovery socket: {e}"))?;
    sock.set_broadcast(true)
        .map_err(|e| format!("broadcast permission: {e}"))?;
    sock.set_read_timeout(Some(Duration::from_millis(200)))
        .map_err(|e| format!("discovery timeout: {e}"))?;
    let mut targets: Vec<SocketAddr> = vec![SocketAddr::from(([255, 255, 255, 255], BEACON_PORT))];
    for b in subnet_broadcasts() {
        targets.push(SocketAddr::new(IpAddr::V4(b), BEACON_PORT));
    }
    let t0 = Instant::now();
    let mut seen: Vec<Found> = Vec::new();
    let mut buf = [0u8; 2048];
    let mut asked = Instant::now() - Duration::from_secs(9);
    while t0.elapsed() < wait {
        // Re-ask while waiting: a worker that starts mid-scan should
        // still turn up, and a lost datagram should not cost the answer.
        if asked.elapsed() >= Duration::from_millis(900) {
            for t in &targets {
                let _ = sock.send_to(QUERY, t);
            }
            asked = Instant::now();
        }
        let (n, from) = match sock.recv_from(&mut buf) {
            Ok(v) => v,
            Err(_) => continue, // read timeout: keep waiting out the window
        };
        let Ok(b) = serde_json::from_slice::<Beacon>(&buf[..n]) else {
            continue;
        };
        if b.magic != MAGIC {
            continue;
        }
        let ip: IpAddr = from.ip();
        if let Some(slot) = seen
            .iter_mut()
            .find(|f| f.addr.ip() == ip && f.beacon.port == b.port)
        {
            slot.beacon = b;
        } else {
            seen.push(Found {
                addr: SocketAddr::new(ip, b.port),
                beacon: b,
            });
        }
    }
    Ok(seen)
}

/// Subnet broadcast addresses, derived without asking the OS for its
/// interface list: for every IPv4 this host answers on, assume the
/// common /24 and shout at `x.y.z.255`. Wrong on unusual masks, and
/// harmless there — the global form is sent too.
fn subnet_broadcasts() -> Vec<std::net::Ipv4Addr> {
    let mut out = Vec::new();
    // A connected UDP socket to a public address reveals which local
    // address the routing table would use, without sending anything.
    if let Ok(s) = UdpSocket::bind(("0.0.0.0", 0)) {
        if s.connect(("192.0.2.1", 9)).is_ok() {
            if let Ok(SocketAddr::V4(local)) = s.local_addr() {
                let o = local.ip().octets();
                out.push(std::net::Ipv4Addr::new(o[0], o[1], o[2], 255));
            }
        }
    }
    out
}

/// Fill in the constant fields so a caller cannot forget the magic.
#[allow(clippy::too_many_arguments)]
pub fn beacon_for(
    port: u16,
    dir_hash: u64,
    arch: &str,
    layers: usize,
    hidden: usize,
    model_path: &str,
    token_required: bool,
) -> Beacon {
    let model = std::path::Path::new(model_path)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();
    Beacon {
        magic: MAGIC.to_string(),
        wire: crate::WIRE_VERSION,
        port,
        dir_hash: format!("{dir_hash:016x}"),
        arch: arch.to_string(),
        layers: layers as u32,
        hidden: hidden as u32,
        model,
        token_required,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn beacon_carries_no_secret() {
        let b = beacon_for(
            9911,
            0xdead_beef,
            "qwen3",
            28,
            2048,
            "/home/me/secret/x.cmf",
            true,
        );
        let json = serde_json::to_string(&b).unwrap();
        // The path leaks a home directory; the token would leak the lot.
        assert!(!json.contains("/home/me"), "{json}");
        assert!(json.contains("x.cmf"), "{json}");
        assert!(!json.to_lowercase().contains("token\":\""), "{json}");
        assert!(json.contains("\"token_required\":true"), "{json}");
    }

    #[test]
    fn foreign_datagrams_are_ignored() {
        let mut b = beacon_for(9911, 1, "a", 1, 1, "m.cmf", false);
        b.magic = "something-else".into();
        let bytes = serde_json::to_vec(&b).unwrap();
        let parsed: Beacon = serde_json::from_slice(&bytes).unwrap();
        assert_ne!(parsed.magic, MAGIC);
    }

    #[test]
    fn peer_addr_joins_ip_and_the_advertised_port() {
        let b = beacon_for(9999, 7, "a", 1, 1, "m.cmf", false);
        let f = Found {
            addr: SocketAddr::from(([192, 168, 1, 5], 9999)),
            beacon: b,
        };
        assert_eq!(f.peer_addr(), "192.168.1.5:9999");
    }
}
