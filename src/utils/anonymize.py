"""
Component-based anonymization utilities.
Splits identifiers into components and hashes each separately for graph construction.
"""
import copy
import hashlib
import hmac
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
IPV4_PATTERN = re.compile(r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$')
IPV6_PATTERN = re.compile(r'^([0-9a-fA-F:]+)::?([0-9a-fA-F:]+)$')

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


def _parse_ipv4(ip: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse IPv4 address and return network and host parts."""
    match = IPV4_PATTERN.match(ip)
    if match:
        network = f"{match.group(1)}.{match.group(2)}.{match.group(3)}"
        host = match.group(4)
        return network, host
    return None, None


def _parse_ipv6(ip: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse IPv6 address and return network and host parts."""
    match = IPV6_PATTERN.match(ip)
    if match:
        network = match.group(1) if match.group(1) else "::"
        host = match.group(2) if match.group(2) else ""
        return network, host
    return None, None


def anonymize_ip(ip: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Hash IP into (full_hash, network_hash, host_hash)."""
    if not ip or ip.strip() == "":
        return None, None, None
    
    ip = ip.strip()
    salt = get_salt("ip")
    full_hash = hash_value(ip, salt)
    
    # Try IPv4 first
    network, host = _parse_ipv4(ip)
    if network and host:
        return full_hash, hash_value(network, salt), hash_value(host, salt)
    
    # Try IPv6
    network, host = _parse_ipv6(ip)
    if network or host:
        network_hash = hash_value(network, salt) if network else None
        host_hash = hash_value(host, salt) if host else None
        return full_hash, network_hash, host_hash
    
    # Unknown format, hash as-is
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


def anonymize_listings_pii(df: pl.DataFrame) -> pl.DataFrame:
    """Anonymize all PII fields in DataFrame after JSON extraction."""
    df_anon = df.clone()
    
    df_anon = _anonymize_email_columns(df_anon)
    df_anon = _anonymize_phone_columns(df_anon)
    df_anon = _anonymize_address_columns(df_anon)
    df_anon = _anonymize_person_name_columns(df_anon)
    
    if "user_ip_address" in df_anon.columns and "user_ip_address_hash" not in df_anon.columns:
        df_anon = _anonymize_ip_column(df_anon)
    
    if "contact_emails" in df_anon.columns and "contact_emails_hash" not in df_anon.columns:
        df_anon = _anonymize_email_list_column(df_anon)
    
    return df_anon
