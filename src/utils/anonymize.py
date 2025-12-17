"""
Component-based anonymization utilities.
Splits identifiers into components and hashes each separately for graph construction.
"""
import copy
import hashlib
import hmac
import ipaddress
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import polars as pl

logger = logging.getLogger(__name__)

# Constants
SALT_PREFIX = "ANON_SALT_"
DEFAULT_SALT_PREFIX = "default_salt_"

# Field name constants
PHONE_FIELDS = ["phone", "mobile"]
BILLING_PHONE_FIELDS = ["phoneDay", "phoneMobile"]
CONTACT_PHONE_FIELDS = ["phone", "mobile"]

# Regex patterns
PHONE_INTERNATIONAL_PATTERN = re.compile(r'^\+(\d{1,3})(\d{2,4})(\d+)$')
PHONE_NATIONAL_PATTERN = re.compile(r'^(\d{2,4})(\d+)$')

# Column name constants for flattened data
EMAIL_COLS = [
    "listing.lister.email",
    "listing.lister.billing.email",
    "listing.lister.contacts.inquiry.email",
    "listing.lister.contacts.viewing.email"
]

PHONE_COLS = [
    "listing.lister.phone",
    "listing.lister.mobile",
    "listing.lister.billing.phoneDay",
    "listing.lister.billing.phoneMobile",
    "listing.lister.contacts.inquiry.phone",
    "listing.lister.contacts.inquiry.mobile",
    "listing.lister.contacts.viewing.phone",
    "listing.lister.contacts.viewing.mobile"
]

ADDRESS_GROUPS = [
    (
        [
            "listing.lister.address.country",
            "listing.lister.address.postalCode",
            "listing.lister.address.locality",
            "listing.lister.address.street"
        ],
        "listing.lister.address"
    ),
    (
        [
            "listing.lister.billing.address.country",
            "listing.lister.billing.address.postalCode",
            "listing.lister.billing.address.locality",
            "listing.lister.billing.address.street"
        ],
        "listing.lister.billing.address"
    ),
    (
        [
            "listing.address.country",
            "listing.address.postalCode",
            "listing.address.locality",
            "listing.address.street"
        ],
        "listing.address"
    ),
]


def get_salt(identifier_type: str) -> bytes:
    """Get salt for identifier type from environment."""
    salt_key = f"{SALT_PREFIX}{identifier_type.upper()}"
    salt = os.getenv(salt_key)
    if not salt:
        logger.warning(f"{salt_key} not set. Using default salt (NOT SECURE).")
        return f"{DEFAULT_SALT_PREFIX}{identifier_type}".encode()
    return salt.encode()


def hash_value(value: str, salt: bytes) -> Optional[str]:
    """Deterministic hash using HMAC-SHA256."""
    if not value or value.strip() == "":
        return None
    return hmac.new(salt, value.strip().lower().encode(), hashlib.sha256).hexdigest()


def anonymize_email(email: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Hash email into (full_hash, user_hash, domain_hash)."""
    if not email or email.strip() == "":
        return None, None, None
    
    email = email.strip().lower()
    
    # Split by @
    if "@" not in email:
        # Invalid email, hash as-is
        salt = get_salt("email")
        full_hash = hash_value(email, salt)
        return full_hash, None, None
    
    parts = email.split("@", 1)
    user_part = parts[0]
    domain_part = parts[1] if len(parts) > 1 else ""
    
    salt = get_salt("email")
    full_hash = hash_value(email, salt)
    user_hash = hash_value(user_part, salt) if user_part else None
    domain_hash = hash_value(domain_part, salt) if domain_part else None
    
    return full_hash, user_hash, domain_hash


def _parse_phone_components(phone: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract country code, area code, and number from phone string."""
    phone = phone.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    
    country_code = None
    area_code = None
    number = None
    
    if phone.startswith("+"):
        # International format
        match = PHONE_INTERNATIONAL_PATTERN.match(phone)
        if match:
            country_code = "+" + match.group(1)
            area_code = match.group(2)
            number = match.group(3)
        else:
            number = phone[1:]
    else:
        # National or local format
        match = PHONE_NATIONAL_PATTERN.match(phone)
        if match:
            area_code = match.group(1)
            number = match.group(2)
        else:
            number = phone
    
    return country_code, area_code, number


def anonymize_phone(phone: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Hash phone into (full_hash, country_hash, area_hash, number_hash)."""
    if not phone or phone.strip() == "":
        return None, None, None, None
    
    phone_clean = phone.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    salt = get_salt("phone")
    full_hash = hash_value(phone_clean, salt)
    
    country_code, area_code, number = _parse_phone_components(phone_clean)
    
    country_hash = hash_value(country_code, salt) if country_code else None
    area_hash = hash_value(area_code, salt) if area_code else None
    number_hash = hash_value(number, salt) if number else None
    
    return full_hash, country_hash, area_hash, number_hash


def _parse_ip_components(ip: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Parse IP address using Python's ipaddress module for robust handling.
    
    Handles:
    - IPv4: Returns /24 network and host octet
    - IPv6: Returns /64 network and interface identifier
    - Zone IDs (e.g., fe80::1%eth0): Stripped before parsing
    
    Args:
        ip: IP address string (IPv4 or IPv6)
        
    Returns:
        Tuple of (network_str, host_str) or (None, None) if invalid
    """
    try:
        # Strip zone ID for link-local addresses (e.g., fe80::1%eth0)
        ip_clean = ip.split('%')[0].strip()
        
        addr = ipaddress.ip_address(ip_clean)
        
        if isinstance(addr, ipaddress.IPv4Address):
            # IPv4: Use /24 network (first 3 octets)
            network = ipaddress.IPv4Network(f"{ip_clean}/24", strict=False)
            network_str = str(network.network_address)
            # Host part is the last octet
            host_str = str(addr).split('.')[-1]
            return network_str, host_str
            
        elif isinstance(addr, ipaddress.IPv6Address):
            # IPv6: Use /64 network (standard subnet size)
            network = ipaddress.IPv6Network(f"{ip_clean}/64", strict=False)
            network_str = str(network.network_address)
            # Host part is the interface identifier (last 64 bits)
            # Represented as the full address for hashing purposes
            host_str = str(addr)
            return network_str, host_str
            
    except ValueError as e:
        logger.debug(f"Failed to parse IP address '{ip}': {e}")
        return None, None
    
    return None, None


def anonymize_ip(ip: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Hash IP into (full_hash, network_hash, host_hash).
    
    Uses Python's ipaddress module for robust parsing of:
    - Standard IPv4 (e.g., 192.168.1.1)
    - Standard IPv6 (e.g., 2001:db8::1)
    - Compressed IPv6 (e.g., ::1, ::ffff:192.0.2.1)
    - Link-local with zone ID (e.g., fe80::1%eth0)
    
    Args:
        ip: IP address string
        
    Returns:
        Tuple of (full_hash, network_hash, host_hash)
    """
    if not ip or ip.strip() == "":
        return None, None, None
    
    ip = ip.strip()
    salt = get_salt("ip")
    full_hash = hash_value(ip, salt)
    
    # Parse IP using robust ipaddress module
    network, host = _parse_ip_components(ip)
    
    if network and host:
        network_hash = hash_value(network, salt)
        host_hash = hash_value(host, salt)
        return full_hash, network_hash, host_hash
    
    # Unknown format, return only full hash
    return full_hash, None, None


def anonymize_address(
    street: str,
    zip_code: str,
    city: str,
    country: str
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Hash address into (full_hash, country_hash, zip_hash, city_hash, street_hash)."""
    salt = get_salt("address")
    
    # Create full address composite (existing format)
    full_address = f"{country}_{zip_code}_{city}_{street}"
    full_hash = hash_value(full_address, salt) if full_address.strip("_") else None
    
    # Hash components separately
    country_hash = hash_value(country, salt) if country else None
    zip_hash = hash_value(zip_code, salt) if zip_code else None
    city_hash = hash_value(city, salt) if city else None
    street_hash = hash_value(street, salt) if street else None
    
    return full_hash, country_hash, zip_hash, city_hash, street_hash


def anonymize_person_name(
    given_name: str,
    family_name: str
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Hash name into (full_hash, given_hash, family_hash)."""
    salt = get_salt("person")
    
    # Create full name (existing format)
    full_name = f"{given_name} {family_name}".strip()
    full_hash = hash_value(full_name, salt) if full_name else None
    
    # Hash components separately
    given_hash = hash_value(given_name, salt) if given_name else None
    family_hash = hash_value(family_name, salt) if family_name else None
    
    return full_hash, given_hash, family_hash


# Polars UDFs for efficient batch processing
def anonymize_email_udf(email: str) -> Dict[str, Optional[str]]:
    """Polars-compatible email anonymization."""
    full, user, domain = anonymize_email(email)
    return {
        "email_hash": full,
        "email_user_hash": user,
        "email_domain_hash": domain
    }


def anonymize_phone_udf(phone: str) -> Dict[str, Optional[str]]:
    """Polars-compatible phone anonymization."""
    full, country, area, number = anonymize_phone(phone)
    return {
        "phone_hash": full,
        "phone_country_hash": country,
        "phone_area_hash": area,
        "phone_number_hash": number
    }


def anonymize_ip_udf(ip: str) -> Dict[str, Optional[str]]:
    """Polars-compatible IP anonymization."""
    full, network, host = anonymize_ip(ip)
    return {
        "ip_hash": full,
        "ip_network_hash": network,
        "ip_host_hash": host
    }


def _anonymize_address_in_dict(addr_dict: Dict, addr_path: List[str]) -> None:
    """Anonymize address fields in a nested dictionary."""
    if not isinstance(addr_dict, dict):
        return
    
    if any(addr_dict.get(k) for k in ["street", "postalCode", "locality", "country"]):
        full_hash, country_hash, zip_hash, city_hash, street_hash = anonymize_address(
            addr_dict.get("street", ""),
            addr_dict.get("postalCode", ""),
            addr_dict.get("locality", ""),
            addr_dict.get("country", "")
        )
        
        # Update dictionary in place
        if street_hash is not None:
            addr_dict["street"] = street_hash
        if zip_hash is not None:
            addr_dict["postalCode"] = zip_hash
        if city_hash is not None:
            addr_dict["locality"] = city_hash
        if country_hash is not None:
            addr_dict["country"] = country_hash


def _anonymize_phone_fields_in_dict(d: Dict, phone_fields: List[str]) -> None:
    """Anonymize phone fields in a dictionary."""
    for phone_field in phone_fields:
        if phone_field in d and d[phone_field]:
            d[phone_field] = anonymize_phone(d[phone_field])[0]


def _anonymize_lister_section(lister: Dict) -> None:
    """Anonymize all PII in the lister section of a listing dictionary."""
    if not isinstance(lister, dict):
        return
    
    # Anonymize email
    if "email" in lister and lister["email"]:
        lister["email"] = anonymize_email(lister["email"])[0]
    
    # Anonymize phones
    _anonymize_phone_fields_in_dict(lister, PHONE_FIELDS)
    
    # Anonymize lister address
    if "address" in lister:
        _anonymize_address_in_dict(lister["address"], ["lister", "address"])
    
    # Anonymize billing section
    if "billing" in lister and isinstance(lister["billing"], dict):
        billing = lister["billing"]
        
        if "email" in billing and billing["email"]:
            billing["email"] = anonymize_email(billing["email"])[0]
        
        _anonymize_phone_fields_in_dict(billing, BILLING_PHONE_FIELDS)
        
        if "address" in billing:
            _anonymize_address_in_dict(billing["address"], ["lister", "billing", "address"])
    
    # Anonymize contacts section
    if "contacts" in lister and isinstance(lister["contacts"], dict):
        contacts = lister["contacts"]
        
        # Anonymize inquiry contact
        if "inquiry" in contacts and isinstance(contacts["inquiry"], dict):
            inquiry = contacts["inquiry"]
            
            # Anonymize names
            if "givenName" in inquiry or "familyName" in inquiry:
                full_hash, given_hash, family_hash = anonymize_person_name(
                    inquiry.get("givenName", ""),
                    inquiry.get("familyName", "")
                )
                if given_hash:
                    inquiry["givenName"] = given_hash
                if family_hash:
                    inquiry["familyName"] = family_hash
            
            # Anonymize email
            if "email" in inquiry and inquiry["email"]:
                inquiry["email"] = anonymize_email(inquiry["email"])[0]
            
            # Anonymize phones
            _anonymize_phone_fields_in_dict(inquiry, CONTACT_PHONE_FIELDS)
        
        # Anonymize viewing contact
        if "viewing" in contacts and isinstance(contacts["viewing"], dict):
            viewing = contacts["viewing"]
            
            if "email" in viewing and viewing["email"]:
                viewing["email"] = anonymize_email(viewing["email"])[0]
            
            _anonymize_phone_fields_in_dict(viewing, CONTACT_PHONE_FIELDS)


def anonymize_listing_json_dict(listing_dict: Dict) -> Dict:
    """Anonymize all PII fields in a listing dictionary."""
    if not listing_dict or not isinstance(listing_dict, dict):
        return listing_dict
    
    anon_dict = copy.deepcopy(listing_dict)
    
    # Anonymize lister section
    if "lister" in anon_dict:
        _anonymize_lister_section(anon_dict["lister"])
    
    # Anonymize property address
    if "address" in anon_dict:
        _anonymize_address_in_dict(anon_dict["address"], ["address"])
    
    return anon_dict


def _anonymize_ip_column(df: pl.DataFrame) -> pl.DataFrame:
    """Anonymize user_ip_address column."""
    if "user_ip_address" not in df.columns:
        return df
    
    def anonymize_ip_wrapper(ip: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Wrapper to handle None values."""
        if ip is None:
            return (None, None, None)
        return anonymize_ip(ip)
    
    ip_hashes = df["user_ip_address"].to_list()
    ip_results = [anonymize_ip_wrapper(ip) for ip in ip_hashes]
    
    df = df.with_columns([
        pl.Series("user_ip_address_hash", [r[0] for r in ip_results], dtype=pl.Utf8),
        pl.Series("user_ip_network_hash", [r[1] for r in ip_results], dtype=pl.Utf8),
        pl.Series("user_ip_host_hash", [r[2] for r in ip_results], dtype=pl.Utf8),
    ])
    return df.drop("user_ip_address")


def _anonymize_email_list_column(df: pl.DataFrame) -> pl.DataFrame:
    """Anonymize contact_emails column (comma-separated list)."""
    if "contact_emails" not in df.columns:
        return df
    
    def anonymize_email_list(email_str: str) -> Optional[str]:
        """Anonymize comma-separated email list."""
        if not email_str:
            return None
        emails = [e.strip() for e in email_str.split(",") if e and e.strip()]
        hashes = [anonymize_email(e)[0] for e in emails if anonymize_email(e)[0]]
        return ",".join(hashes) if hashes else None
    
    email_hashes = df["contact_emails"].to_list()
    email_results = [anonymize_email_list(emails) for emails in email_hashes]
    
    df = df.with_columns([
        pl.Series("contact_emails_hash", email_results, dtype=pl.Utf8)
    ])
    return df.drop("contact_emails")


def _anonymize_listing_json_column(df: pl.DataFrame) -> pl.DataFrame:
    """Anonymize listing_json column by parsing, anonymizing, and re-serializing."""
    if "listing_json" not in df.columns:
        return df
    
    def anonymize_json_string(json_str: str) -> str:
        """Parse JSON, anonymize PII, and re-serialize."""
        if not json_str:
            return json_str
        
        try:
            listing_dict = json.loads(json_str)
            anon_dict = anonymize_listing_json_dict(listing_dict)
            return json.dumps(anon_dict, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError, AttributeError) as e:
            logger.warning(f"Failed to parse/anonymize listing_json: {e}")
            return json_str
    
    json_strings = df["listing_json"].to_list()
    anon_json_strings = [anonymize_json_string(js) for js in json_strings]
    
    return df.with_columns([
        pl.Series("listing_json", anon_json_strings, dtype=pl.Utf8)
    ])


def anonymize_dataframe_batch(df: pl.DataFrame) -> pl.DataFrame:
    """Anonymize PII columns in DataFrame including embedded PII in listing_json."""
    df_anon = df.clone()
    
    df_anon = _anonymize_ip_column(df_anon)
    df_anon = _anonymize_email_list_column(df_anon)
    df_anon = _anonymize_listing_json_column(df_anon)
    
    return df_anon


def _anonymize_email_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Anonymize email columns in flattened DataFrame."""
    df_anon = df.clone()
    
    for col in EMAIL_COLS:
        if col in df_anon.columns:
            emails = df_anon[col].to_list()
            email_results = [anonymize_email(e) if e else (None, None, None) for e in emails]
            
            df_anon = df_anon.with_columns([
                pl.Series(f"{col}.hash", [r[0] for r in email_results], dtype=pl.Utf8),
                pl.Series(f"{col}.user_hash", [r[1] for r in email_results], dtype=pl.Utf8),
                pl.Series(f"{col}.domain_hash", [r[2] for r in email_results], dtype=pl.Utf8),
            ])
            df_anon = df_anon.drop(col)
    
    return df_anon


def _anonymize_phone_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Anonymize phone columns in flattened DataFrame."""
    df_anon = df.clone()
    
    for col in PHONE_COLS:
        if col in df_anon.columns:
            phones = df_anon[col].to_list()
            phone_results = [anonymize_phone(p) if p else (None, None, None, None) for p in phones]
            
            df_anon = df_anon.with_columns([
                pl.Series(f"{col}.hash", [r[0] for r in phone_results], dtype=pl.Utf8),
                pl.Series(f"{col}.country_hash", [r[1] for r in phone_results], dtype=pl.Utf8),
                pl.Series(f"{col}.area_hash", [r[2] for r in phone_results], dtype=pl.Utf8),
                pl.Series(f"{col}.number_hash", [r[3] for r in phone_results], dtype=pl.Utf8),
            ])
            df_anon = df_anon.drop(col)
    
    return df_anon


def _anonymize_address_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Anonymize address columns in flattened DataFrame."""
    df_anon = df.clone()
    
    for cols, base_path in ADDRESS_GROUPS:
        if all(col in df_anon.columns for col in cols):
            country_col, zip_col, city_col, street_col = cols
            
            countries = df_anon[country_col].fill_null("").to_list()
            zips = df_anon[zip_col].fill_null("").to_list()
            cities = df_anon[city_col].fill_null("").to_list()
            streets = df_anon[street_col].fill_null("").to_list()
            
            address_hashes = [
                anonymize_address(street, zip_code, city, country)
                for street, zip_code, city, country in zip(streets, zips, cities, countries)
            ]
            
            df_anon = df_anon.with_columns([
                pl.Series(f"{base_path}.address_hash", [h[0] for h in address_hashes], dtype=pl.Utf8),
                pl.Series(f"{base_path}.country_hash", [h[1] for h in address_hashes], dtype=pl.Utf8),
                pl.Series(f"{base_path}.zip_hash", [h[2] for h in address_hashes], dtype=pl.Utf8),
                pl.Series(f"{base_path}.city_hash", [h[3] for h in address_hashes], dtype=pl.Utf8),
                pl.Series(f"{base_path}.street_hash", [h[4] for h in address_hashes], dtype=pl.Utf8),
            ])
            df_anon = df_anon.drop(cols)
    
    return df_anon


def _anonymize_person_name_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Anonymize person name columns in flattened DataFrame."""
    df_anon = df.clone()
    
    given_col = "listing.lister.contacts.inquiry.givenName"
    family_col = "listing.lister.contacts.inquiry.familyName"
    
    if given_col in df_anon.columns and family_col in df_anon.columns:
        given_names = df_anon[given_col].to_list()
        family_names = df_anon[family_col].to_list()
        name_hashes = [
            anonymize_person_name(g, f)[0] if (g or f) else None
            for g, f in zip(given_names, family_names)
        ]
        
        df_anon = df_anon.with_columns([
            pl.Series("listing.lister.contacts.inquiry.name_hash", name_hashes, dtype=pl.Utf8)
        ])
        df_anon = df_anon.drop([given_col, family_col])
    
    return df_anon


def _anonymize_owner_id_column(df: pl.DataFrame) -> pl.DataFrame:
    """Anonymize owner_id column - this is a direct user identifier."""
    if "owner_id" not in df.columns:
        return df
    
    salt = get_salt("owner")
    owner_ids = df["owner_id"].to_list()
    hashes = [hash_value(str(oid), salt) if oid else None for oid in owner_ids]
    
    df = df.with_columns([
        pl.Series("owner_id_hash", hashes, dtype=pl.Utf8)
    ])
    return df.drop("owner_id")


def _anonymize_object_reference_column(df: pl.DataFrame) -> pl.DataFrame:
    """Anonymize object_reference column - may contain identifiable patterns."""
    if "object_reference" not in df.columns:
        return df
    
    salt = get_salt("object")
    refs = df["object_reference"].to_list()
    hashes = [hash_value(str(ref), salt) if ref else None for ref in refs]
    
    df = df.with_columns([
        pl.Series("object_reference_hash", hashes, dtype=pl.Utf8)
    ])
    return df.drop("object_reference")


def anonymize_listings_pii(df: pl.DataFrame) -> pl.DataFrame:
    """Anonymize all PII fields in DataFrame after JSON extraction."""
    df_anon = df.clone()
    
    # Anonymize structured PII columns
    df_anon = _anonymize_email_columns(df_anon)
    df_anon = _anonymize_phone_columns(df_anon)
    df_anon = _anonymize_address_columns(df_anon)
    df_anon = _anonymize_person_name_columns(df_anon)
    
    # Anonymize direct identifiers
    df_anon = _anonymize_owner_id_column(df_anon)
    df_anon = _anonymize_object_reference_column(df_anon)
    
    if "user_ip_address" in df_anon.columns and "user_ip_address_hash" not in df_anon.columns:
        df_anon = _anonymize_ip_column(df_anon)
    
    if "contact_emails" in df_anon.columns and "contact_emails_hash" not in df_anon.columns:
        df_anon = _anonymize_email_list_column(df_anon)
    
    return df_anon


# =============================================================================
# NEW: Anonymization for merged SEON + Events data
# =============================================================================

# PII columns from Events (flattened Snowflake data)
EVENTS_PII_PATTERNS = {
    "email": [
        "LISTING_LISTER_EMAIL",
        "LISTING_LISTER_BILLING_EMAIL",
        "LISTING_LISTER_CONTACTS_INQUIRY_EMAIL",
        "LISTING_LISTER_CONTACTS_VIEWING_EMAIL",
        "CREATEDATUSERNAME",  # Contains email in format timestamp#email
    ],
    "phone": [
        "LISTING_LISTER_PHONE",
        "LISTING_LISTER_MOBILE",
        "LISTING_LISTER_BILLING_PHONEDAY",
        "LISTING_LISTER_BILLING_PHONEMOBILE",
        "LISTING_LISTER_CONTACTS_INQUIRY_PHONE",
        "LISTING_LISTER_CONTACTS_INQUIRY_MOBILE",
        "LISTING_LISTER_CONTACTS_VIEWING_PHONE",
        "LISTING_LISTER_CONTACTS_VIEWING_MOBILE",
    ],
    "name": [
        "LISTING_LISTER_NAME",
        "LISTING_LISTER_LEGALNAME",
        "LISTING_LISTER_BILLING_NAME",
        "LISTING_LISTER_BILLING_COMPANYNAME",
        "LISTING_LISTER_CONTACTS_INQUIRY_GIVENNAME",
        "LISTING_LISTER_CONTACTS_INQUIRY_FAMILYNAME",
        "LISTING_LISTER_USERNAME",
    ],
    "address": [
        "LISTING_ADDRESS_STREET",
        "LISTING_ADDRESS_LOCALITY",
        "LISTING_ADDRESS_POSTALCODE",
        "LISTING_LISTER_ADDRESS_STREET",
        "LISTING_LISTER_ADDRESS_LOCALITY",
        "LISTING_LISTER_ADDRESS_POSTALCODE",
        "LISTING_LISTER_BILLING_ADDRESS_STREET",
        "LISTING_LISTER_BILLING_ADDRESS_LOCALITY",
        "LISTING_LISTER_BILLING_ADDRESS_POSTALCODE",
    ],
    "geo": [
        "LISTING_ADDRESS_GEOCOORDINATES_LATITUDE",
        "LISTING_ADDRESS_GEOCOORDINATES_LONGITUDE",
    ],
}

# PII columns from SEON
SEON_PII_PATTERNS = {
    "email": [
        "email/email",
        "email/raw_email",
        "user_name",  # Often contains email
    ],
    "phone": [
        "phone_number",
        "billing_phone",
        "bin_phone",
    ],
    "name": [
        "user_fullname",
    ],
    "address": [
        "billing_street",
        "billing_city",
        "billing_zip",
    ],
    "ip": [
        "ip",
        "session/device_ip_address",
        "session/dns_ip",
    ],
    "user_id": [
        "user_id",
    ],
}

# Events IP column (needs special handling to create graph-compatible names)
EVENTS_IP_COLUMN = "USERIPADDRESS"


def _anonymize_column_generic(
    df: pl.DataFrame,
    col: str,
    salt_type: str,
    hash_suffix: str = "_hash",
) -> pl.DataFrame:
    """Anonymize a single column with generic hashing."""
    if col not in df.columns:
        return df
    
    salt = get_salt(salt_type)
    values = df[col].to_list()
    hashes = [hash_value(str(v), salt) if v is not None else None for v in values]
    
    df = df.with_columns([
        pl.Series(f"{col}{hash_suffix}", hashes, dtype=pl.Utf8)
    ])
    return df.drop(col)


def _anonymize_geo_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Anonymize geo coordinates by rounding to reduce precision.
    This preserves approximate location (city-level) while removing exact address.
    """
    geo_cols = EVENTS_PII_PATTERNS.get("geo", [])
    
    for col in geo_cols:
        if col in df.columns:
            # Round to 2 decimal places (~1km precision)
            df = df.with_columns(
                pl.col(col).cast(pl.Float64, strict=False).round(2).alias(col)
            )
    
    return df


def _anonymize_createdatusername(df: pl.DataFrame) -> pl.DataFrame:
    """
    Anonymize CREATEDATUSERNAME which contains timestamp#email format.
    Extract and hash just the email part.
    """
    col = "CREATEDATUSERNAME"
    if col not in df.columns:
        return df
    
    salt = get_salt("email")
    values = df[col].to_list()
    
    def extract_and_hash_email(val):
        if val is None:
            return None
        # Format: 2025-07-24T10:15:40.218729Z#email@domain.com
        if "#" in str(val):
            parts = str(val).split("#", 1)
            if len(parts) > 1:
                email = parts[1]
                return hash_value(email, salt)
        return hash_value(str(val), salt)
    
    hashes = [extract_and_hash_email(v) for v in values]
    
    df = df.with_columns([
        pl.Series(f"{col}_hash", hashes, dtype=pl.Utf8)
    ])
    return df.drop(col)


def anonymize_merged_data(df: pl.DataFrame) -> pl.DataFrame:
    """
    Anonymize all PII in merged SEON + Events DataFrame.
    
    This handles:
    - Events PII: emails, phones, names, addresses from LISTING_* columns
    - SEON PII: emails, phones, IPs, user IDs from SEON columns
    - Geo coordinates: rounded for privacy
    
    Args:
        df: Merged DataFrame with both events and SEON columns
        
    Returns:
        Anonymized DataFrame
    """
    logger.info("Anonymizing merged data...")
    df_anon = df.clone()
    
    # Track columns anonymized
    anonymized_count = 0
    
    # Handle special case: CREATEDATUSERNAME
    if "CREATEDATUSERNAME" in df_anon.columns:
        df_anon = _anonymize_createdatusername(df_anon)
        anonymized_count += 1
    
    # Anonymize Events email columns
    for col in EVENTS_PII_PATTERNS.get("email", []):
        if col in df_anon.columns and col != "CREATEDATUSERNAME":
            df_anon = _anonymize_column_generic(df_anon, col, "email")
            anonymized_count += 1
    
    # Anonymize Events phone columns
    for col in EVENTS_PII_PATTERNS.get("phone", []):
        if col in df_anon.columns:
            df_anon = _anonymize_column_generic(df_anon, col, "phone")
            anonymized_count += 1
    
    # Anonymize Events name columns
    for col in EVENTS_PII_PATTERNS.get("name", []):
        if col in df_anon.columns:
            df_anon = _anonymize_column_generic(df_anon, col, "person")
            anonymized_count += 1
    
    # Anonymize Events address columns
    for col in EVENTS_PII_PATTERNS.get("address", []):
        if col in df_anon.columns:
            df_anon = _anonymize_column_generic(df_anon, col, "address")
            anonymized_count += 1
    
    # Round geo coordinates
    df_anon = _anonymize_geo_columns(df_anon)
    
    # Anonymize SEON email columns
    for col in SEON_PII_PATTERNS.get("email", []):
        if col in df_anon.columns:
            df_anon = _anonymize_column_generic(df_anon, col, "email")
            anonymized_count += 1
    
    # Anonymize SEON phone columns
    for col in SEON_PII_PATTERNS.get("phone", []):
        if col in df_anon.columns:
            df_anon = _anonymize_column_generic(df_anon, col, "phone")
            anonymized_count += 1
    
    # Anonymize SEON name columns
    for col in SEON_PII_PATTERNS.get("name", []):
        if col in df_anon.columns:
            df_anon = _anonymize_column_generic(df_anon, col, "person")
            anonymized_count += 1
    
    # Anonymize SEON address columns
    for col in SEON_PII_PATTERNS.get("address", []):
        if col in df_anon.columns:
            df_anon = _anonymize_column_generic(df_anon, col, "address")
            anonymized_count += 1
    
    # Anonymize SEON IP columns
    for col in SEON_PII_PATTERNS.get("ip", []):
        if col in df_anon.columns:
            df_anon = _anonymize_column_generic(df_anon, col, "ip")
            anonymized_count += 1
    
    # Anonymize Events IP column (USERIPADDRESS) with graph-compatible output names
    if EVENTS_IP_COLUMN in df_anon.columns:
        ip_values = df_anon[EVENTS_IP_COLUMN].to_list()
        ip_results = [anonymize_ip(ip) if ip else (None, None, None) for ip in ip_values]
        
        df_anon = df_anon.with_columns([
            pl.Series("ip_hash", [r[0] for r in ip_results], dtype=pl.Utf8),
            pl.Series("ip_network_hash", [r[1] for r in ip_results], dtype=pl.Utf8),
        ])
        df_anon = df_anon.drop(EVENTS_IP_COLUMN)
        anonymized_count += 1
        logger.debug(f"  Created ip_hash and ip_network_hash from {EVENTS_IP_COLUMN}")
    
    # Anonymize SEON user_id columns
    for col in SEON_PII_PATTERNS.get("user_id", []):
        if col in df_anon.columns:
            df_anon = _anonymize_column_generic(df_anon, col, "owner")
            anonymized_count += 1
    
    # Anonymize INSERTION_ID (contains user-generated reference)
    if "INSERTION_ID" in df_anon.columns:
        df_anon = _anonymize_column_generic(df_anon, "INSERTION_ID", "object")
        anonymized_count += 1
    
    # Anonymize OBJECTREFERENCE
    if "OBJECTREFERENCE" in df_anon.columns:
        df_anon = _anonymize_column_generic(df_anon, "OBJECTREFERENCE", "object")
        anonymized_count += 1
    
    logger.info(f"  Anonymized {anonymized_count} PII columns")
    
    return df_anon
