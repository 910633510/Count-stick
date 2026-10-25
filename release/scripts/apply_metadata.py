#!/usr/bin/env python3
"""Prepare the NEW x.Count-stick listing through Apple's official API.

Run with ~/.appstoreconnect/venv/bin/python. Default: read-only dry run.
  apply_metadata.py --validate-only
  apply_metadata.py --app-id NEW_ID [--set-price]
  apply_metadata.py --app-id NEW_ID --privacy-policy-url HTTPS_URL --apply [--set-price]

Does not create an app record, upload/select builds or assets, change availability,
publish privacy answers, or submit for review. --set-price uses CHN CNY 12 as the
base price; Apple calculates other storefront prices. Custom schedules are refused.
"""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = "x.Count-stick"
SKU = "count-stick-20261022"
LOCALES = ("en-US", "zh-Hans")
EDITABLE = {"PREPARE_FOR_SUBMISSION", "REJECTED", "METADATA_REJECTED", "DEVELOPER_REJECTED"}
PRIVATE = Path.home() / ".codex/app-store-connect-release/runs/count-stick-20261022"
REDACT_KEYS = {"contactFirstName", "contactLastName", "contactPhone", "contactEmail",
               "demoAccountName", "demoAccountPassword", "password", "token",
               "authorization", "uploadOperations"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relation(kind, resource_id):
    return {"data": {"type": kind, "id": resource_id}}


def body(kind, attributes=None, resource_id=None, relationships=None):
    data = {"type": kind}
    if attributes is not None:
        data["attributes"] = attributes
    if resource_id is not None:
        data["id"] = resource_id
    if relationships is not None:
        data["relationships"] = relationships
    return {"data": data}


def validate_shape(value, schema, spec, location="request"):
    """Check the OpenAPI request subset used here, without third-party packages."""
    if "$ref" in schema:
        schema = spec["components"]["schemas"][schema["$ref"].rsplit("/", 1)[1]]
    if value is None:
        require(schema.get("nullable", False), location + " cannot be null")
        return
    if "oneOf" in schema:
        for option in schema["oneOf"]:
            try:
                validate_shape(value, option, spec, location)
                return
            except ValueError:
                pass
        raise ValueError(location + " matches no request variant")
    expected = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int}
    if expected in types:
        require(isinstance(value, types[expected]), location + " has incorrect type")
    if "enum" in schema:
        require(value in schema["enum"], location + " has unsupported value")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            require(key in value, location + "." + key + " is required")
        for key, item in value.items():
            require(key in schema.get("properties", {}), location + "." + key + " is unsupported")
            validate_shape(item, schema["properties"][key], spec, location + "." + key)
    elif isinstance(value, list):
        for item in value:
            validate_shape(item, schema["items"], spec, location + "[]")


def safe_url(value):
    parsed = urlparse(value or "")
    require(parsed.scheme == "https" and parsed.hostname and not parsed.username
            and not parsed.password, "A public HTTPS URL is required")
    return value


class Release:
    def __init__(self, args):
        self.args = args
        self.metadata = json.loads(args.metadata.read_text())
        self.spec = json.loads(args.openapi.read_text())
        self.operations = []
        self.sequence = 0
        self.replacements = {}
        self.contact_values = []
        require(self.metadata["bundleId"] == BUNDLE, "Metadata targets another bundle")
        require(self.metadata["appRecord"]["sku"] == SKU, "Unexpected new-app SKU")
        require(set(self.metadata["localizations"]) == set(LOCALES), "Only en-US and zh-Hans are supported")
        require(self.metadata["monetization"]["type"] == "PAID_UPFRONT", "Expected paid-upfront metadata")
        require(self.metadata["monetization"]["requestedBaseTerritory"] == "CHN"
                and self.metadata["monetization"]["currency"] == "CNY"
                and Decimal(self.metadata["monetization"]["customerPrice"]) == 12,
                "This helper only supports the requested China CNY 12 price")
        for locale in LOCALES:
            entry = self.metadata["localizations"][locale]
            a, v = entry["appInfo"], entry["version"]
            require(0 < len(a["name"]) <= 30 and len(a["subtitle"]) <= 30, locale + " name/subtitle too long")
            require(len(v["keywords"].encode("utf-8")) <= 100, locale + " keywords exceed 100 bytes")
            require(len(v["description"]) <= 4000 and len(v["promotionalText"]) <= 170, locale + " copy too long")
        self.privacy_url = args.privacy_policy_url or self.metadata["shared"].get("privacyPolicyUrl")
        if self.privacy_url:
            safe_url(self.privacy_url)
        for key in ("supportUrl", "marketingUrl"):
            safe_url(self.metadata["shared"][key])
        require(len(self.metadata["reviewNotesDraft"].encode("utf-8")) <= 4000, "Review notes exceed 4000 bytes")
        require(not args.apply or self.privacy_url, "Set a verified published --privacy-policy-url before --apply")
        self.run_dir = PRIVATE / ("metadata-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        self.run_dir.mkdir(mode=0o700, parents=True)
        self.run_dir.chmod(0o700)

    def sanitize(self, value):
        if isinstance(value, dict):
            return {k: "[REDACTED]" if k.lower() in {s.lower() for s in REDACT_KEYS}
                    else self.sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.sanitize(v) for v in value]
        if isinstance(value, str):
            for contact in self.contact_values:
                value = value.replace(contact, "[REDACTED]")
        return value

    def save(self, name, data):
        path = self.run_dir / name
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(self.sanitize(data), handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def connect(self):
        require(self.args.app_id and re.fullmatch(r"[0-9]+", self.args.app_id), "--app-id must be the new numeric App Store ID")
        require(self.args.app_id != "6752362515", "The old StickCount app is outside this task")
        client = Path.home() / ".appstoreconnect/asc_api.py"
        module_spec = importlib.util.spec_from_file_location("count_stick_asc", client)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        self.raw_request = module.request
        self.verify_target()

    def verify_target(self):
        app = self.raw_request("GET", "/v1/apps/" + self.args.app_id)["data"]
        require(app["attributes"]["bundleId"] == BUNDLE, "Live app ID does not match x.Count-stick")
        require(app["attributes"]["sku"] == SKU, "Live app SKU does not match the new app handoff")
        return app

    def request(self, method, path, payload=None, optional=False):
        # Reverify identity before every operation, including paginated reads.
        self.verify_target()
        parsed = urlparse(path)
        require(not parsed.scheme or (parsed.scheme == "https" and parsed.netloc == "api.appstoreconnect.apple.com"), "Unexpected API pagination host")
        require(method == "GET" or self.args.apply, "Writes require --apply")
        self.sequence += 1
        try:
            response = self.raw_request(method, path, payload)
        except Exception as exc:
            status = re.search(r"HTTP (\d{3})", str(exc))
            details = None
            if status:
                try:
                    details = json.loads(str(exc).split(": ", 1)[1]).get("errors")
                except (ValueError, IndexError):
                    pass
            self.save(f"{self.sequence:03d}-error.json", {"method": method, "path": path,
                      "errorType": type(exc).__name__, "httpStatus": status.group(1) if status else None,
                      "errors": details})
            if optional and status and status.group(1) == "404":
                return {"data": None}
            raise RuntimeError("API request failed; inspect the operation and private receipt. No automatic write retry was attempted.") from None
        self.save(f"{self.sequence:03d}-response.json", {"method": method, "path": path, "response": response})
        return response

    def all(self, path):
        data, included = [], []
        while path:
            response = self.request("GET", path)
            data.extend(response.get("data", []))
            included.extend(response.get("included", []))
            path = response.get("links", {}).get("next")
        return data, included

    def add(self, method, path, payload, schema, label, category=False):
        validate_shape(payload, self.spec["components"]["schemas"][schema], self.spec)
        self.operations.append({"method": method, "path": path, "payload": payload,
                                "schema": schema, "label": label, "category": category})

    def upsert(self, kind, current, wanted, relationship, parent_kind, parent_id, schema_stem, label):
        if current:
            changed = {k: v for k, v in wanted.items() if k != "locale" and current.get("attributes", {}).get(k) != v}
            if changed:
                self.add("PATCH", "/v1/" + kind + "/" + current["id"], body(kind, changed, current["id"]), schema_stem + "UpdateRequest", label)
        else:
            self.add("POST", "/v1/" + kind, body(kind, wanted, relationships={relationship: relation(parent_kind, parent_id)}), schema_stem + "CreateRequest", label)

    def contact(self):
        profile = json.loads((Path.home() / ".codex/app-store-connect-release/profile.json").read_text())
        contact = profile["publisher"]["default_review_contact"]
        mapping = {"first_name": "contactFirstName", "last_name": "contactLastName", "phone": "contactPhone", "email": "contactEmail"}
        values = {target: contact[source] for source, target in mapping.items()}
        require(all(isinstance(v, str) and v for v in values.values()), "Incomplete private review contact profile")
        self.contact_values = list(values.values())
        return values

    def plan(self):
        app_id = self.args.app_id
        infos, _ = self.all(f"/v1/apps/{app_id}/appInfos?limit=200")
        editable_infos = [d for d in infos if d.get("attributes", {}).get("state", d.get("attributes", {}).get("appStoreState")) in EDITABLE]
        require(len(editable_infos) == 1, "Expected one editable appInfo for the new app")
        info = editable_infos[0]
        versions, _ = self.all(f"/v1/apps/{app_id}/appStoreVersions?limit=200")
        versions = [d for d in versions if d["attributes"]["platform"] == "IOS"]
        require(len(versions) <= 1, "This helper only prepares the new app's first iOS version")
        version = versions[0] if versions else None
        if version:
            a = version["attributes"]
            require(a["versionString"] == self.args.version, "Existing first version differs from --version")
            require(a.get("appVersionState", a.get("appStoreState")) in EDITABLE, "Version is not editable; no review state changes are made")
        version_id = version["id"] if version else "__NEW_VERSION__"
        if not version:
            self.add("POST", "/v1/appStoreVersions", body("appStoreVersions", {"platform": "IOS", "versionString": self.args.version, "copyright": self.metadata["shared"]["copyright"]}, relationships={"app": relation("apps", app_id)}), "AppStoreVersionCreateRequest", "Create first iOS version")
        elif version["attributes"].get("copyright") != self.metadata["shared"]["copyright"]:
            self.add("PATCH", "/v1/appStoreVersions/" + version_id, body("appStoreVersions", {"copyright": self.metadata["shared"]["copyright"]}, version_id), "AppStoreVersionUpdateRequest", "Version copyright")
        info_locs, _ = self.all(f"/v1/appInfos/{info['id']}/appInfoLocalizations?limit=200")
        version_locs, _ = self.all(f"/v1/appStoreVersions/{version_id}/appStoreVersionLocalizations?limit=200") if version else ([], [])
        for locale in LOCALES:
            source = self.metadata["localizations"][locale]
            wanted = {**source["appInfo"], "locale": locale}
            if self.privacy_url:
                wanted["privacyPolicyUrl"] = self.privacy_url
            found = [d for d in info_locs if d["attributes"]["locale"] == locale]
            require(len(found) <= 1, "Duplicate appInfo localization")
            self.upsert("appInfoLocalizations", found[0] if found else None, wanted, "appInfo", "appInfos", info["id"], "AppInfoLocalization", locale + " app information")
            wanted = {k: v for k, v in source["version"].items() if v is not None and k != "whatsNew"}
            wanted.update({"locale": locale, "supportUrl": self.metadata["shared"]["supportUrl"], "marketingUrl": self.metadata["shared"]["marketingUrl"]})
            found = [d for d in version_locs if d["attributes"]["locale"] == locale]
            require(len(found) <= 1, "Duplicate version localization")
            self.upsert("appStoreVersionLocalizations", found[0] if found else None, wanted, "appStoreVersion", "appStoreVersions", version_id, "AppStoreVersionLocalization", locale + " store copy")
        category = self.request("GET", f"/v1/appInfos/{info['id']}/primaryCategory", optional=True).get("data")
        wanted_category = self.metadata["shared"]["primaryCategory"]
        if not category or category["id"] != wanted_category:
            self.add("PATCH", "/v1/appInfos/" + info["id"], body("appInfos", resource_id=info["id"], relationships={"primaryCategory": relation("appCategories", wanted_category)}), "AppInfoUpdateRequest", "Primary category", category=True)
        review = self.request("GET", f"/v1/appStoreVersions/{version_id}/appStoreReviewDetail", optional=True).get("data") if version else None
        wanted = {**self.contact(), "demoAccountRequired": False, "notes": self.metadata["reviewNotesDraft"]}
        self.upsert("appStoreReviewDetails", review, wanted, "appStoreVersion", "appStoreVersions", version_id, "AppStoreReviewDetail", "Review details")
        if self.args.set_price:
            self.plan_price()

    def price_state(self):
        schedule = self.request("GET", f"/v1/apps/{self.args.app_id}/appPriceSchedule?include=baseTerritory", optional=True).get("data")
        if not schedule:
            return None, [], {}
        rows, included = self.all(f"/v1/appPriceSchedules/{schedule['id']}/manualPrices?include=appPricePoint,territory&limit=200")
        points = {d["id"]: d for d in included if d["type"] == "appPricePoints"}
        return schedule, rows, points

    def plan_price(self):
        points, _ = self.all(f"/v1/apps/{self.args.app_id}/appPricePoints?filter[territory]=CHN&include=territory&limit=200")
        matches = [d for d in points if Decimal(d["attributes"]["customerPrice"]) == Decimal("12") and d.get("relationships", {}).get("territory", {}).get("data", {}).get("id") == "CHN"]
        require(len(matches) == 1, "Expected exactly one live CHN CNY 12 price point")
        schedule, rows, current_points = self.price_state()
        if schedule and schedule.get("relationships", {}).get("baseTerritory", {}).get("data", {}).get("id") == "CHN" and len(rows) == 1 and rows[0].get("relationships", {}).get("appPricePoint", {}).get("data", {}).get("id") == matches[0]["id"] and not rows[0]["attributes"].get("startDate") and not rows[0]["attributes"].get("endDate"):
            return
        require(len(rows) <= 1, "Refusing to replace a custom multi-entry price schedule")
        for row in rows:
            a = row["attributes"]
            point = current_points[row["relationships"]["appPricePoint"]["data"]["id"]]
            require(not a.get("startDate") and not a.get("endDate") and Decimal(point["attributes"]["customerPrice"]) == 0, "Refusing to replace an existing paid or scheduled price")
        payload = body("appPriceSchedules", relationships={"app": relation("apps", self.args.app_id), "baseTerritory": relation("territories", "CHN"), "manualPrices": {"data": [{"type": "appPrices", "id": "${price1}"}]}})
        payload["included"] = [{"type": "appPrices", "id": "${price1}", "attributes": {"startDate": None, "endDate": None}, "relationships": {"appPricePoint": relation("appPricePoints", matches[0]["id"])}}]
        self.add("POST", "/v1/appPriceSchedules", payload, "AppPriceScheduleCreateRequest", "CNY 12 base price; Apple equalizes other storefronts")

    def apply(self):
        for operation in self.operations:
            current = json.loads(json.dumps(operation).replace("__NEW_VERSION__", self.replacements.get("__NEW_VERSION__", "__NEW_VERSION__")))
            payload = current["payload"]
            require("__NEW_VERSION__" not in json.dumps(payload), "Missing newly created version ID")
            response = self.request(current["method"], current["path"], payload)["data"]
            if current["path"] == "/v1/appStoreVersions":
                self.replacements["__NEW_VERSION__"] = response["id"]
            if current["path"] == "/v1/appPriceSchedules":
                schedule, rows, points = self.price_state()
                require(schedule["relationships"]["baseTerritory"]["data"]["id"] == "CHN", "Price base-territory readback differs")
                valid = [r for r in rows if r["relationships"]["territory"]["data"]["id"] == "CHN" and Decimal(points[r["relationships"]["appPricePoint"]["data"]["id"]]["attributes"]["customerPrice"]) == 12 and not r["attributes"].get("startDate") and not r["attributes"].get("endDate")]
                require(len(valid) == 1, "CNY 12 immediate price readback failed")
            elif current["category"]:
                category = self.request("GET", f"/v1/appInfos/{response['id']}/primaryCategory")["data"]
                require(category["id"] == payload["data"]["relationships"]["primaryCategory"]["data"]["id"], "Category readback differs")
            else:
                readback = self.request("GET", f"/v1/{response['type']}/{response['id']}")["data"]
                require(all(readback.get("attributes", {}).get(k) == v for k, v in payload["data"].get("attributes", {}).items()), "Persisted metadata differs from intended values")
            print("Verified:", operation["label"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app-id", help="The NEW x.Count-stick App Store ID")
    parser.add_argument("--apply", action="store_true", help="Perform and re-read the planned mutations")
    parser.add_argument("--set-price", action="store_true", help="Also prepare/apply CNY 12 as the base price")
    parser.add_argument("--validate-only", action="store_true", help="Offline draft and request-shape checks; no API calls")
    parser.add_argument("--privacy-policy-url", help="Already published and verified app-specific HTTPS policy")
    parser.add_argument("--version", default="1.0")
    parser.add_argument("--metadata", type=Path, default=ROOT / "release/metadata-draft.json")
    parser.add_argument("--openapi", type=Path, default=ROOT / "output/apple-openapi.json")
    args = parser.parse_args()
    require(not (args.validate_only and args.apply), "--validate-only cannot be combined with --apply")
    release = Release(args)
    if args.validate_only:
        # Exercise create and update request shapes without importing credentials.
        for locale in LOCALES:
            e = release.metadata["localizations"][locale]
            info_attributes = {**e["appInfo"], "locale": locale}
            if release.privacy_url:
                info_attributes["privacyPolicyUrl"] = release.privacy_url
            version_attributes = {k: v for k, v in {**e["version"], "locale": locale}.items() if v is not None}
            version_attributes.update({key: release.metadata["shared"][key] for key in ("supportUrl", "marketingUrl")})
            for existing in (None, {"id": "sample", "attributes": {}}):
                release.upsert("appInfoLocalizations", existing, info_attributes, "appInfo", "appInfos", "sample", "AppInfoLocalization", locale)
                release.upsert("appStoreVersionLocalizations", existing, version_attributes, "appStoreVersion", "appStoreVersions", "sample", "AppStoreVersionLocalization", locale)
                release.upsert("appStoreReviewDetails", existing, {"contactFirstName": "Example", "contactLastName": "Reviewer", "contactPhone": "+10000000000", "contactEmail": "review@example.com", "demoAccountRequired": False, "notes": release.metadata["reviewNotesDraft"]}, "appStoreVersion", "appStoreVersions", "sample", "AppStoreReviewDetail", "Review fixture")
        release.add("POST", "/v1/appStoreVersions", body("appStoreVersions", {"platform": "IOS", "versionString": args.version, "copyright": release.metadata["shared"]["copyright"]}, relationships={"app": relation("apps", "sample")}), "AppStoreVersionCreateRequest", "First-version fixture")
        release.add("PATCH", "/v1/appStoreVersions/sample", body("appStoreVersions", {"copyright": release.metadata["shared"]["copyright"]}, "sample"), "AppStoreVersionUpdateRequest", "Copyright fixture")
        release.add("PATCH", "/v1/appInfos/sample", body("appInfos", resource_id="sample", relationships={"primaryCategory": relation("appCategories", "UTILITIES")}), "AppInfoUpdateRequest", "Category fixture")
        price_payload = body("appPriceSchedules", relationships={"app": relation("apps", "sample"), "baseTerritory": relation("territories", "CHN"), "manualPrices": {"data": [{"type": "appPrices", "id": "${price1}"}]}})
        price_payload["included"] = [{"type": "appPrices", "id": "${price1}", "attributes": {"startDate": None, "endDate": None}, "relationships": {"appPricePoint": relation("appPricePoints", "live-point-placeholder")}}]
        release.add("POST", "/v1/appPriceSchedules", price_payload, "AppPriceScheduleCreateRequest", "Price fixture")
        release.save("offline-validation.json", {"openapiVersion": release.spec["info"]["version"], "bundleId": BUNDLE, "validatedOperations": release.operations})
        print("PASS: locale lengths and all 10 used OpenAPI request schema variants; no API calls.")
    else:
        release.connect()
        release.plan()
        release.save("plan.json", {"mode": "APPLY" if args.apply else "DRY_RUN", "bundleId": BUNDLE, "appId": args.app_id, "operations": release.operations})
        for operation in release.operations:
            print(operation["method"], operation["path"], "—", operation["label"])
        if not release.privacy_url:
            print("Privacy policy URL is missing; apply is blocked until supplied.")
        if args.apply:
            release.apply()
            release.save("completed.json", {"bundleId": BUNDLE, "appId": args.app_id, "verifiedOperationCount": len(release.operations), "submitted": False})
        else:
            print("DRY RUN: no mutations performed. Add --apply only after reviewing the private plan.")
    print("Private receipts:", release.run_dir)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print("STOP:", str(exc), file=sys.stderr)
        sys.exit(1)
