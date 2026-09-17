import asyncio
import base64
import hashlib
import ipaddress
import sqlite3
import time
import unittest

from pool import AddressDriver, ForeignAddressError, Pool, PoolError


class FakeDriver:
    prefix = ipaddress.IPv6Network("2606:4700:1234:abcd::/80")

    def __init__(self):
        self.owned = {}
        self.removed = []
        self.fail_probe = False

    async def addresses(self):
        return self.owned.copy()

    async def add(self, address):
        self.owned[address] = {"prefixlen": 128}

    async def remove(self, address):
        self.removed.append(address)
        self.owned.pop(address, None)

    async def probe(self, address):
        if self.fail_probe:
            raise PoolError("probe failed")


class PoolTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.driver = FakeDriver()
        self.pool = Pool({"public_url": "https://pool.example:18443", "api_token": "a" * 48,
                          "ready_target": 3, "batch_size": 3}, self.driver, sqlite3.connect(":memory:"))

    async def asyncTearDown(self):
        await self.pool.close()

    async def test_batch_prepares_distinct_addresses(self):
        await asyncio.gather(*(self.pool.generate_one() for attempt in range(8)))
        leases = [self.pool.allocate() for attempt in range(8)]
        self.assertEqual(len({lease["ipv6"] for lease in leases}), 8)
        self.assertEqual(self.pool.status()["counts"]["leased"], 8)
        with self.assertRaises(PoolError):
            self.pool.allocate()

    async def test_release_never_recycles_address_or_credentials(self):
        await self.pool.generate_one()
        lease = self.pool.allocate()
        from urllib.parse import urlsplit
        url = urlsplit(lease["proxy_url"])
        header = "Basic " + base64.b64encode(f"{url.username}:{url.password}".encode()).decode()
        self.assertEqual(self.pool.authorize_proxy(header)["address"], lease["ipv6"])
        await self.pool.release(lease["id"])
        await self.pool.release(lease["id"])
        with self.assertRaises(PoolError):
            self.pool.authorize_proxy(header)
        await self.pool.generate_one()
        self.assertNotEqual(self.pool.allocate()["ipv6"], lease["ipv6"])
        self.assertIn(lease["ipv6"], self.driver.removed)

    async def test_expired_credentials_rejected(self):
        await self.pool.generate_one()
        lease = self.pool.allocate()
        self.pool.update("UPDATE addresses SET expires=? WHERE lease=?", (time.time() - 1, lease["id"]))
        from urllib.parse import urlsplit
        url = urlsplit(lease["proxy_url"])
        header = "Basic " + base64.b64encode(f"{url.username}:{url.password}".encode()).decode()
        with self.assertRaises(PoolError):
            self.pool.authorize_proxy(header)

    async def test_failed_candidates_not_allocated(self):
        self.driver.fail_probe = True
        with self.assertRaises(PoolError):
            await self.pool.generate_one()
        with self.assertRaises(PoolError):
            self.pool.allocate()
        await self.pool.cleanup()
        self.assertEqual(len(self.driver.owned), 0)

    async def test_cleanup_preserves_foreign_address(self):
        foreign = "2606:4700:1234:abcd:ffff::1"
        self.driver.owned[foreign] = {"prefixlen": 128}
        await self.pool.generate_one()
        lease = self.pool.allocate()
        await self.pool.release(lease["id"])
        self.assertIn(foreign, self.driver.owned)
        self.assertNotIn(foreign, self.driver.removed)

    async def test_restart_invalidates_leases(self):
        await self.pool.generate_one()
        lease = self.pool.allocate()
        await self.pool.start()
        row = self.pool.query("SELECT status FROM addresses WHERE lease=?", (lease["id"],)).fetchone()
        self.assertEqual(row["status"], "retired")

    async def test_unknown_and_malformed_auth_rejected(self):
        for authorization in ("", "Bearer secret", "Basic !", "Basic " + base64.b64encode(b"a:b").decode()):
            with self.assertRaises(PoolError):
                self.pool.authorize_proxy(authorization)

    async def test_active_limit(self):
        self.pool.config["max_leases"] = 1
        await self.pool.generate_one()
        self.pool.allocate()
        await self.pool.generate_one()
        with self.assertRaises(PoolError) as error:
            self.pool.allocate()
        self.assertEqual(error.exception.status, 429)

    async def test_foreign_collision_is_not_cleaned_up(self):
        async def reject(address):
            self.driver.owned[address] = {"prefixlen": 128}
            raise ForeignAddressError("foreign")
        self.driver.add = reject
        with self.assertRaises(ForeignAddressError):
            await self.pool.generate_one()
        await self.pool.cleanup()
        self.assertEqual(len(self.driver.owned), 1)
        self.assertEqual(self.driver.removed, [])

    async def test_linux_boolean_tentative_flag_waits_for_readiness(self):
        driver = AddressDriver("eth0", "2606:4700:1234:abcd::/80")
        address = "2606:4700:1234:abcd::1234"
        snapshots = [{}, {address: {"tentative": True}}, {address: {"prefixlen": 128}}]
        calls = []

        async def addresses():
            calls.append(True)
            return snapshots.pop(0)

        async def command(*args):
            return b""

        driver.addresses = addresses
        driver.command = command
        await driver.add(address)
        self.assertEqual(len(calls), 3)


if __name__ == "__main__":
    unittest.main()
