import argparse
import asyncio
import base64
import contextlib
import hashlib
import hmac
import ipaddress
import json
import logging
import os
from pathlib import Path
import secrets
import signal
import socket
import sqlite3
import ssl
import time
from urllib.parse import urlsplit


class PoolError(Exception):
    def __init__(self, message, status=503):
        super().__init__(message)
        self.status = status


class ForeignAddressError(PoolError):
    pass


class AddressDriver:
    def __init__(self, interface, prefix):
        self.interface = interface
        self.prefix = ipaddress.IPv6Network(prefix)
        if self.prefix.prefixlen < 64 or self.prefix.prefixlen > 96:
            raise ValueError("prefix must be a dedicated /64 through /96 subnet")
        if not self.prefix.network_address.is_global:
            raise ValueError("a globally routed prefix is required")

    async def command(self, *args):
        proc = await asyncio.create_subprocess_exec(
            "ip", "-6", *args, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), 10)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            raise PoolError("address command timeout") from None
        except asyncio.CancelledError:
            if proc.returncode is None:
                proc.kill()
            await proc.wait()
            raise
        if proc.returncode:
            raise PoolError("address command failed: " + stderr.decode()[:180])
        return stdout

    async def addresses(self):
        result = json.loads(await self.command("-j", "addr", "show", "dev", self.interface))
        return {entry["local"]: entry for device in result for entry in device.get("addr_info", [])}

    async def add(self, address):
        if address in await self.addresses():
            raise ForeignAddressError("refusing to take ownership of an existing address")
        try:
            await self.command("addr", "add", address + "/128", "dev", self.interface,
                               "preferred_lft", "0", "noprefixroute")
        except PoolError as error:
            if error.status == 503:
                logging.warning("pool operation unavailable: %s", str(error)[:180])
            if "File exists" in str(error):
                raise ForeignAddressError("address was concurrently claimed") from None
            raise
        for attempt in range(40):
            entry = (await self.addresses()).get(address)
            flags = entry.get("flags", []) if entry else []
            if entry and (entry.get("dadfailed") or "dadfailed" in flags):
                raise PoolError("duplicate address detected")
            if entry and not entry.get("tentative") and "tentative" not in flags:
                return
            await asyncio.sleep(.1)
        raise PoolError("address readiness timeout")

    async def remove(self, address):
        if ipaddress.IPv6Address(address) not in self.prefix:
            raise PoolError("refusing cleanup outside owned subnet")
        current = (await self.addresses()).get(address)
        if current:
            if current.get("prefixlen") != 128:
                raise PoolError("refusing cleanup of non-host address")
            await self.command("addr", "del", address + "/128", "dev", self.interface)

    async def connect(self, host, port, source):
        loop = asyncio.get_running_loop()
        addresses = await asyncio.wait_for(loop.getaddrinfo(
            host, port, family=socket.AF_INET6, type=socket.SOCK_STREAM), 8)
        last_error = None
        for address in dict.fromkeys(item[4] for item in addresses):
            if not ipaddress.IPv6Address(address[0]).is_global:
                continue
            sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
            sock.setblocking(False)
            try:
                sock.bind((source, 0, 0, 0))
                await asyncio.wait_for(loop.sock_connect(sock, address), 6)
                return await asyncio.open_connection(sock=sock)
            except (OSError, asyncio.TimeoutError) as error:
                last_error = error
                sock.close()
            except asyncio.CancelledError:
                sock.close()
                raise
        raise PoolError("IPv6 target connection failed") from last_error

    async def probe(self, address):
        reader, writer = await self.connect("chatgpt.com", 443, address)
        writer.close()
        await writer.wait_closed()


class Pool:
    def __init__(self, config, driver, database):
        self.config = config
        self.driver = driver
        self.database = database
        self.database.row_factory = sqlite3.Row
        self.database.execute("PRAGMA journal_mode=WAL")
        self.database.execute("CREATE TABLE IF NOT EXISTS addresses ("
                              "address TEXT PRIMARY KEY, status TEXT NOT NULL, "
                              "lease TEXT UNIQUE, password_hash TEXT, expires REAL, "
                              "created REAL NOT NULL)")
        self.database.commit()
        self.connections = {}
        self.generating = False
        self.wake = asyncio.Event()
        self.closed = False
        self.last_error = ""
        self.refill_task = None
        self.sweep_task = None
        self.network_slots = asyncio.Semaphore(config.get("max_connections", 64))

    def query(self, sql, args=()):
        return self.database.execute(sql, args)

    def update(self, sql, args=()):
        self.query(sql, args)
        self.database.commit()

    async def start(self):
        self.update("UPDATE addresses SET status='cleanup' WHERE status IN ('leased','adding')")
        await self.cleanup()
        for row in self.query("SELECT address FROM addresses WHERE status='ready'").fetchall():
            try:
                if row["address"] not in await self.driver.addresses():
                    await self.driver.add(row["address"])
                await self.driver.probe(row["address"])
            except Exception:
                self.update("UPDATE addresses SET status='cleanup' WHERE address=?", (row["address"],))
        self.refill_task = asyncio.create_task(self.refill_loop())
        self.sweep_task = asyncio.create_task(self.sweep_loop())
        self.wake.set()

    async def cleanup(self):
        for row in self.query("SELECT address FROM addresses WHERE status='cleanup'").fetchall():
            try:
                await self.driver.remove(row["address"])
                self.update("UPDATE addresses SET status='retired', password_hash=NULL WHERE address=?", (row["address"],))
            except Exception:
                self.last_error = "address cleanup failed"
                logging.exception("owned address cleanup failed")

    async def generate_one(self):
        current = await self.driver.addresses()
        address = None
        for attempt in range(20):
            candidate = str(ipaddress.IPv6Address(
                int(self.driver.prefix.network_address) | secrets.randbits(128 - self.driver.prefix.prefixlen)))
            if candidate not in current and not self.query("SELECT 1 FROM addresses WHERE address=?", (candidate,)).fetchone():
                address = candidate
                break
        if address is None:
            raise PoolError("cannot select an unused address")
        self.update("INSERT INTO addresses(address,status,created) VALUES (?,'adding',?)", (address, time.time()))
        try:
            await self.driver.add(address)
            await self.driver.probe(address)
            self.update("UPDATE addresses SET status='ready' WHERE address=?", (address,))
        except ForeignAddressError:
            self.update("UPDATE addresses SET status='foreign' WHERE address=?", (address,))
            raise
        except Exception:
            self.update("UPDATE addresses SET status='cleanup' WHERE address=?", (address,))
            raise

    async def refill_loop(self):
        while not self.closed:
            await self.wake.wait()
            self.wake.clear()
            self.generating = True
            try:
                while not self.closed:
                    count = self.query("SELECT COUNT(*) FROM addresses WHERE status='ready'").fetchone()[0]
                    needed = self.config.get("ready_target", 24) - count
                    if needed <= 0:
                        break
                    results = await asyncio.gather(*(
                        self.generate_one() for attempt in range(min(needed, self.config.get("batch_size", 6)))
                    ), return_exceptions=True)
                    failures = [result for result in results if isinstance(result, BaseException)]
                    if failures:
                        self.last_error = "some IPv6 candidates failed readiness checks"
                        logging.warning("batch generation: %d failed candidates", len(failures))
                        logging.warning("candidate failure category: %s", str(failures[0])[:200])
                        await self.cleanup()
                        await asyncio.sleep(5)
                    else:
                        self.last_error = ""
            finally:
                self.generating = False

    async def sweep_loop(self):
        while not self.closed:
            await asyncio.sleep(5)
            expired = self.query("SELECT lease FROM addresses WHERE status='leased' AND expires<=?", (time.time(),)).fetchall()
            for row in expired:
                await self.release(row["lease"])
            await self.cleanup()
            self.wake.set()

    def allocate(self):
        active = self.query("SELECT COUNT(*) FROM addresses WHERE status='leased'").fetchone()[0]
        if active >= self.config.get("max_leases", 32):
            raise PoolError("active lease limit reached", 429)
        row = self.query("SELECT address FROM addresses WHERE status='ready' ORDER BY created LIMIT 1").fetchone()
        if row is None:
            self.wake.set()
            raise PoolError("pool warming or exhausted")
        lease = secrets.token_hex(16)
        password = secrets.token_urlsafe(32)
        expires = time.time() + self.config.get("lease_seconds", 120)
        self.update("UPDATE addresses SET status='leased', lease=?, password_hash=?, expires=? WHERE address=?",
                    (lease, hashlib.sha256(password.encode()).hexdigest(), expires, row["address"]))
        self.wake.set()
        endpoint = urlsplit(self.config["public_url"])
        return {"id": lease, "ipv6": row["address"], "expires_at": int(expires),
                "proxy_url": f"https://{lease}:{password}@{endpoint.netloc}"}

    def authorize_proxy(self, authorization):
        if not authorization.startswith("Basic "):
            raise PoolError("proxy authentication required", 407)
        try:
            username, password = base64.b64decode(authorization[6:], validate=True).decode().split(":", 1)
        except (ValueError, UnicodeError):
            raise PoolError("invalid proxy credentials", 407) from None
        row = self.query("SELECT * FROM addresses WHERE lease=? AND status='leased'", (username,)).fetchone()
        digest = hashlib.sha256(password.encode()).hexdigest()
        if row is None or row["expires"] <= time.time() or not hmac.compare_digest(row["password_hash"] or "", digest):
            raise PoolError("expired or invalid proxy lease", 407)
        return row

    async def release(self, lease):
        row = self.query("SELECT address,status FROM addresses WHERE lease=?", (lease,)).fetchone()
        if row is None:
            raise PoolError("unknown lease", 404)
        if row["status"] in ("cleanup", "retired"):
            return
        self.update("UPDATE addresses SET status='cleanup',password_hash=NULL WHERE lease=?", (lease,))
        for writer in list(self.connections.pop(lease, set())):
            writer.close()
        await self.cleanup()
        self.wake.set()

    def status(self):
        counts = {row["status"]: row["count"] for row in self.query("SELECT status,COUNT(*) AS count FROM addresses GROUP BY status")}
        return {"counts": counts, "generating": self.generating,
                "active_connections": sum(len(value) for value in self.connections.values()),
                "last_error": self.last_error}

    async def respond(self, writer, status, payload):
        encoded = json.dumps(payload).encode()
        phrase = {200: "OK", 201: "Created", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 405: "Method Not Allowed", 407: "Proxy Authentication Required", 429: "Too Many Requests", 503: "Service Unavailable"}.get(status, "Error")
        writer.write((f"HTTP/1.1 {status} {phrase}\r\nContent-Type: application/json\r\n"
                      f"Content-Length: {len(encoded)}\r\nConnection: close\r\nCache-Control: no-store\r\n\r\n").encode() + encoded)
        await writer.drain()

    async def tunnel(self, reader, writer, target, headers):
        if target not in self.config.get("allowed_targets", ["chatgpt.com:443"]):
            raise PoolError("target is not allowed", 400)
        row = self.authorize_proxy(headers.get("proxy-authorization", ""))
        lease = row["lease"]
        if len(self.connections.get(lease, set())) >= 2:
            raise PoolError("lease connection limit reached", 429)
        if self.network_slots.locked():
            raise PoolError("connection limit reached", 429)
        host, port = target.rsplit(":", 1)
        self.connections.setdefault(lease, set()).add(writer)
        upstream_writer = None
        try:
            async with self.network_slots:
                upstream_reader, upstream_writer = await asyncio.wait_for(
                    self.driver.connect(host, int(port), row["address"]), 25)
                current = self.query("SELECT status,expires FROM addresses WHERE lease=?", (lease,)).fetchone()
                if not current or current["status"] != "leased" or current["expires"] <= time.time():
                    raise PoolError("lease released during connection", 407)
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()

                async def copy(source, destination):
                    while True:
                        data = await asyncio.wait_for(source.read(65536), 45)
                        if not data:
                            break
                        destination.write(data)
                        await destination.drain()

                tasks = [asyncio.create_task(copy(reader, upstream_writer)),
                         asyncio.create_task(copy(upstream_reader, writer))]
                try:
                    await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED,
                                       timeout=max(0, row["expires"] - time.time()))
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            self.connections.get(lease, set()).discard(writer)
            if not self.connections.get(lease):
                self.connections.pop(lease, None)
            if upstream_writer:
                upstream_writer.close()
                with contextlib.suppress(Exception):
                    await upstream_writer.wait_closed()

    async def handle(self, reader, writer):
        try:
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
            if len(raw) > 16384:
                raise PoolError("headers too large", 400)
            lines = raw.decode("ascii").split("\r\n")
            method, target, version = lines[0].split(" ")
            if version != "HTTP/1.1":
                raise PoolError("HTTP/1.1 required", 400)
            headers = {}
            for line in lines[1:]:
                if not line:
                    continue
                name, value = line.split(":", 1)
                key = name.strip().lower()
                if key in headers:
                    raise PoolError("duplicate header", 400)
                headers[key] = value.strip()
            if headers.get("transfer-encoding") or headers.get("content-length", "0") != "0":
                raise PoolError("request body is not accepted", 400)
            if method == "CONNECT":
                await self.tunnel(reader, writer, target, headers)
                return
            expected = "Bearer " + self.config["api_token"]
            if not hmac.compare_digest(headers.get("authorization", ""), expected):
                raise PoolError("authentication required", 401)
            if method == "GET" and target == "/v1/status":
                await self.respond(writer, 200, self.status())
            elif method == "POST" and target == "/v1/leases":
                await self.respond(writer, 201, self.allocate())
            elif method == "DELETE" and target.startswith("/v1/leases/"):
                await self.release(target[len("/v1/leases/"):])
                await self.respond(writer, 200, {"released": True})
            else:
                raise PoolError("unknown endpoint", 404)
        except PoolError as error:
            with contextlib.suppress(Exception):
                await self.respond(writer, error.status, {"error": str(error)})
        except (ValueError, UnicodeError, asyncio.IncompleteReadError, asyncio.LimitOverrunError):
            with contextlib.suppress(Exception):
                await self.respond(writer, 400, {"error": "invalid request"})
        except (OSError, asyncio.TimeoutError):
            logging.warning("proxy connection timed out or failed")
            with contextlib.suppress(Exception):
                await self.respond(writer, 503, {"error": "connection failed"})
        except Exception:
            logging.exception("request failed")
        finally:
            writer.close()
            with contextlib.suppress(Exception):
                await writer.wait_closed()

    async def close(self):
        self.closed = True
        for task in (self.refill_task, self.sweep_task):
            if task:
                task.cancel()
        await asyncio.gather(*(task for task in (self.refill_task, self.sweep_task) if task), return_exceptions=True)
        for row in self.query("SELECT lease FROM addresses WHERE status='leased'").fetchall():
            await self.release(row["lease"])
        self.update("UPDATE addresses SET status='cleanup' WHERE status IN ('ready','adding')")
        await self.cleanup()
        self.database.close()


async def main(config_file):
    config = json.loads(Path(config_file).read_text())
    if len(config.get("api_token", "")) < 32:
        raise ValueError("api_token must have at least 32 characters")
    endpoint = urlsplit(config["public_url"])
    if endpoint.scheme != "https" or not endpoint.hostname or endpoint.username:
        raise ValueError("public_url must be an HTTPS origin")
    directory = Path(config.get("data_dir", "/var/lib/sub2api-turn-state-egress"))
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    database = sqlite3.connect(directory / "pool.sqlite3")
    driver = AddressDriver(config["interface"], config["prefix"])
    pool = Pool(config, driver, database)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.minimum_version = ssl.TLSVersion.TLSv1_2
    tls.load_cert_chain(config["tls_cert"], config["tls_key"])
    tls.set_alpn_protocols(["http/1.1"])
    server = await asyncio.start_server(pool.handle, config.get("listen", "0.0.0.0"),
                                        config.get("port", 18443), ssl=tls, limit=16384,
                                        ssl_handshake_timeout=10)
    await pool.start()
    stopped = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, stopped.set)
    logging.info("independent turn-state egress pool started")
    try:
        await stopped.wait()
    finally:
        server.close()
        await server.wait_closed()
        await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    options = parser.parse_args()
    os.umask(0o077)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main(options.config))
