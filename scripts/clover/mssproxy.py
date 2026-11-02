#!/usr/bin/env python3
# Local HTTP(S) proxy for a pod whose path drops packets above ~1240 bytes:
# containers cannot set the MTU or advmss, but a socket can advertise a small MSS.
import socket, threading
MSS = 1200
def out(host, port):
    for fam, typ, pro, _, addr in socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM):
        s = socket.socket(fam, typ, pro)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_MAXSEG, MSS)
        s.settimeout(30)
        s.connect(addr)
        s.settimeout(None)
        return s
    raise OSError("no address")
def pump(a, b):
    try:
        while True:
            d = a.recv(262144)
            if not d: break
            b.sendall(d)
    except OSError: pass
    for x in (a, b):
        try: x.shutdown(socket.SHUT_RDWR)
        except OSError: pass
def handle(c):
    try:
        buf = b""
        while b"\r\n\r\n" not in buf:
            d = c.recv(65536)
            if not d: return c.close()
            buf += d
        line, rest = buf.split(b"\r\n", 1)
        method, target, ver = line.split(b" ", 2)
        if method == b"CONNECT":
            host, port = target.decode().rsplit(":", 1)
            s = out(host, int(port))
            c.sendall(b"HTTP/1.1 200 OK\r\n\r\n")
        else:
            t = target.decode()
            assert t.startswith("http://")
            hp, _, path = t[7:].partition("/")
            host, _, port = hp.partition(":")
            s = out(host, int(port or 80))
            s.sendall(method + b" /" + path.encode() + b" " + ver + b"\r\n" + rest)
        threading.Thread(target=pump, args=(s, c), daemon=True).start()
        pump(c, s)
    except Exception:
        try: c.close()
        except OSError: pass
l = socket.socket(); l.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
l.bind(("127.0.0.1", 3128)); l.listen(256)
while True:
    c, _ = l.accept()
    threading.Thread(target=handle, args=(c,), daemon=True).start()
