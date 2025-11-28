"""
Component-Based Anonymization Utilities

This module provides anonymization functions that split identifiers into components
and hash each component separately. This approach:
1. Preserves graph structure (deterministic hashing)
2. Creates additional features (component-level connections)
3. Potentially improves performance by enabling more granular pattern detection
4. Maintains privacy (cannot reverse hashes)

Example:
    Email: "user@domain.com" → hash("user") + "@" + hash("domain.com")
    This allows tracking:
    - User-part reuse across domains (fraud pattern)
    - Domain-part reuse across users (fraud pattern)
    - Full email reuse (existing pattern)
"""
import hashlib
import hmac
import os
import re
from typing import Optional, Tuple
import polars as pl


# Per-type salts (should be stored in environment variables, not in code)
def get_salt(identifier_type: str) -> bytes:
    """Get salt for identifier type from environment."""
    salt_key = f"ANON_SALT_{identifier_type.upper()}"
    salt = os.getenv(salt_key)
    if not salt:
        # Fallback: use a default salt (NOT RECOMMENDED for production)
        # In production, always set environment variables
        print(f"Warning: {salt_key} not set. Using default salt (NOT SECURE).")
        return f"default_salt_{identifier_type}".encode()
    return salt.encode()


def hash_value(value: str, salt: bytes) -> str:
    """Deterministic hash using HMAC-SHA256."""
    if not value or value.strip() == "":
        return None
    return hmac.new(salt, value.strip().lower().encode(), hashlib.sha256).hexdigest()


def anonymize_email(email: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Split email into user-part and domain-part, hash each separately.
    
    Returns:
        (full_hash, user_hash, domain_hash)
        - full_hash: Hash of entire email (for backward compatibility)
        - user_hash: Hash of user part (before @)
        - domain_hash: Hash of domain part (after @)
    
    Example:
        "user@domain.com" → (
            hash("user@domain.com"),  # Full email hash
            hash("user"),             # User-part hash
            hash("domain.com")        # Domain-part hash
        )
    """
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
    
    # Hash full email (for backward compatibility)
    full_hash = hash_value(email, salt)
    
    # Hash user part separately
    user_hash = hash_value(user_part, salt) if user_part else None
    
    # Hash domain part separately
    domain_hash = hash_value(domain_part, salt) if domain_part else None
    
    return full_hash, user_hash, domain_hash


def anonymize_phone(phone: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Split phone into country code, area code, and number, hash each separately.
    
    Phone format assumptions:
    - International: +41 44 123 4567 or +41441234567
    - National: 044 123 4567 or 0441234567
    - Local: 123 4567 or 1234567
    
    Returns:
        (full_hash, country_hash, area_hash, number_hash)
        - full_hash: Hash of entire phone (for backward compatibility)
        - country_hash: Hash of country code (if present)
        - area_hash: Hash of area code (if present)
        - number_hash: Hash of local number
    
    Example:
        "+41 44 123 4567" → (
            hash("+41 44 123 4567"),  # Full phone hash
            hash("+41"),               # Country code hash
            hash("44"),                # Area code hash
            hash("1234567")            # Number hash
        )
    """
    if not phone or phone.strip() == "":
        return None, None, None, None
    
    phone = phone.strip().replace(" ", "").replace("-", "").replace("(", "").replace(")", "")
    
    salt = get_salt("phone")
    
    # Hash full phone (for backward compatibility)
    full_hash = hash_value(phone, salt)
    
    # Try to extract country code (starts with +)
    country_code = None
    area_code = None
    number = None
    
    if phone.startswith("+"):
        # International format: +[country][area][number]
        # Try to extract country code (1-3 digits after +)
        match = re.match(r'^\+(\d{1,3})(\d{2,4})(\d+)$', phone)
        if match:
            country_code = "+" + match.group(1)
            area_code = match.group(2)
            number = match.group(3)
        else:
            # Fallback: treat everything after + as number
            number = phone[1:]
    else:
        # National or local format
        # Try to extract area code (2-4 digits at start)
        match = re.match(r'^(\d{2,4})(\d+)$', phone)
        if match:
            area_code = match.group(1)
            number = match.group(2)
        else:
            # No area code, treat as local number
            number = phone
    
    # Hash components
    country_hash = hash_value(country_code, salt) if country_code else None
    area_hash = hash_value(area_code, salt) if area_code else None
    number_hash = hash_value(number, salt) if number else None
    
    return full_hash, country_hash, area_hash, number_hash


def anonymize_ip(ip: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Split IP address into network and host parts, hash each separately.
    
    For IPv4: Split by subnet (e.g., /24 network)
    For IPv6: Split by prefix
    
    Returns:
        (full_hash, network_hash, host_hash)
        - full_hash: Hash of entire IP (for backward compatibility)
        - network_hash: Hash of network part (e.g., 192.168.1.*)
        - host_hash: Hash of host part (last octet for IPv4)
    
    Example:
        "192.168.1.100" → (
            hash("192.168.1.100"),  # Full IP hash
            hash("192.168.1"),      # Network hash (/24)
            hash("100")             # Host hash
        )
    """
    if not ip or ip.strip() == "":
        return None, None, None
    
    ip = ip.strip()
    salt = get_salt("ip")
    
    # Hash full IP (for backward compatibility)
    full_hash = hash_value(ip, salt)
    
    # Try to parse IPv4
    ipv4_match = re.match(r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$', ip)
    if ipv4_match:
        # Use /24 network (first 3 octets)
        network = f"{ipv4_match.group(1)}.{ipv4_match.group(2)}.{ipv4_match.group(3)}"
        host = ipv4_match.group(4)
        network_hash = hash_value(network, salt)
        host_hash = hash_value(host, salt)
        return full_hash, network_hash, host_hash
    
    # Try to parse IPv6 (simplified: use first 64 bits as network)
    ipv6_match = re.match(r'^([0-9a-fA-F:]+)::?([0-9a-fA-F:]+)$', ip)
    if ipv6_match:
        # Use first part as network
        network = ipv6_match.group(1) if ipv6_match.group(1) else "::"
        host = ipv6_match.group(2) if ipv6_match.group(2) else ""
        network_hash = hash_value(network, salt) if network else None
        host_hash = hash_value(host, salt) if host else None
        return full_hash, network_hash, host_hash
    
    # Unknown format, hash as-is
    return full_hash, None, None


def anonymize_address(street: str, zip_code: str, city: str, country: str) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    """
    Hash address components separately.
    
    Returns:
        (full_hash, country_hash, zip_hash, city_hash, street_hash)
        - full_hash: Hash of full address composite (for backward compatibility)
        - country_hash: Hash of country
        - zip_hash: Hash of postal code
        - city_hash: Hash of city
        - street_hash: Hash of street
    
    Note: This preserves the existing address_id format while adding component-level hashing.
    """
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


def anonymize_person_name(given_name: str, family_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Hash person name components separately.
    
    Returns:
        (full_hash, given_hash, family_hash)
        - full_hash: Hash of full name (for backward compatibility)
        - given_hash: Hash of given name
        - family_hash: Hash of family name
    """
    salt = get_salt("person")
    
    # Create full name (existing format)
    full_name = f"{given_name} {family_name}".strip()
    full_hash = hash_value(full_name, salt) if full_name else None
    
    # Hash components separately
    given_hash = hash_value(given_name, salt) if given_name else None
    family_hash = hash_value(family_name, salt) if family_name else None
    
    return full_hash, given_hash, family_hash


# Polars UDFs for efficient batch processing
def anonymize_email_udf(email: str) -> dict:
    """Polars-compatible email anonymization."""
    full, user, domain = anonymize_email(email)
    return {
        "email_hash": full,
        "email_user_hash": user,
        "email_domain_hash": domain
    }


def anonymize_phone_udf(phone: str) -> dict:
    """Polars-compatible phone anonymization."""
    full, country, area, number = anonymize_phone(phone)
    return {
        "phone_hash": full,
        "phone_country_hash": country,
        "phone_area_hash": area,
        "phone_number_hash": number
    }


def anonymize_ip_udf(ip: str) -> dict:
    """Polars-compatible IP anonymization."""
    full, network, host = anonymize_ip(ip)
    return {
        "ip_hash": full,
        "ip_network_hash": network,
        "ip_host_hash": host
    }


def anonymize_listing_json_dict(listing_dict: dict) -> dict:
    """
    Anonymizes all PII in a parsed listing JSON dictionary.
    
    This function recursively processes the listing structure and replaces
    all PII fields (emails, phones, addresses, names) with their hashed equivalents.
    
    Args:
        listing_dict: Parsed listing JSON as a dictionary
        
    Returns:
        Dictionary with all PII fields anonymized
    """
    import copy
    
    if not listing_dict or not isinstance(listing_dict, dict):
        return listing_dict
    
    anon_dict = copy.deepcopy(listing_dict)
    
    # Anonymize lister section
    if "lister" in anon_dict and isinstance(anon_dict["lister"], dict):
        lister = anon_dict["lister"]
        
        # Anonymize email
        if "email" in lister and lister["email"]:
            anon_dict["lister"]["email"] = anonymize_email(lister["email"])[0]
        
        # Anonymize phones
        for phone_field in ["phone", "mobile"]:
            if phone_field in lister and lister[phone_field]:
                anon_dict["lister"][phone_field] = anonymize_phone(lister[phone_field])[0]
        
        # Anonymize lister address
        if "address" in lister and isinstance(lister["address"], dict):
            addr = lister["address"]
            if addr.get("street") or addr.get("postalCode") or addr.get("locality") or addr.get("country"):
                full_hash, country_hash, zip_hash, city_hash, street_hash = anonymize_address(
                    addr.get("street", ""),
                    addr.get("postalCode", ""),
                    addr.get("locality", ""),
                    addr.get("country", "")
                )
                anon_dict["lister"]["address"]["street"] = street_hash
                anon_dict["lister"]["address"]["postalCode"] = zip_hash
                anon_dict["lister"]["address"]["locality"] = city_hash
                anon_dict["lister"]["address"]["country"] = country_hash
        
        # Anonymize billing section
        if "billing" in lister and isinstance(lister["billing"], dict):
            billing = lister["billing"]
            
            # Anonymize billing email
            if "email" in billing and billing["email"]:
                anon_dict["lister"]["billing"]["email"] = anonymize_email(billing["email"])[0]
            
            # Anonymize billing phones
            for phone_field in ["phoneDay", "phoneMobile"]:
                if phone_field in billing and billing[phone_field]:
                    anon_dict["lister"]["billing"][phone_field] = anonymize_phone(billing[phone_field])[0]
            
            # Anonymize billing address
            if "address" in billing and isinstance(billing["address"], dict):
                addr = billing["address"]
                if addr.get("street") or addr.get("postalCode") or addr.get("locality") or addr.get("country"):
                    full_hash, country_hash, zip_hash, city_hash, street_hash = anonymize_address(
                        addr.get("street", ""),
                        addr.get("postalCode", ""),
                        addr.get("locality", ""),
                        addr.get("country", "")
                    )
                    anon_dict["lister"]["billing"]["address"]["street"] = street_hash
                    anon_dict["lister"]["billing"]["address"]["postalCode"] = zip_hash
                    anon_dict["lister"]["billing"]["address"]["locality"] = city_hash
                    anon_dict["lister"]["billing"]["address"]["country"] = country_hash
        
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
                        anon_dict["lister"]["contacts"]["inquiry"]["givenName"] = given_hash
                    if family_hash:
                        anon_dict["lister"]["contacts"]["inquiry"]["familyName"] = family_hash
                
                # Anonymize email
                if "email" in inquiry and inquiry["email"]:
                    anon_dict["lister"]["contacts"]["inquiry"]["email"] = anonymize_email(inquiry["email"])[0]
                
                # Anonymize phones
                for phone_field in ["phone", "mobile"]:
                    if phone_field in inquiry and inquiry[phone_field]:
                        anon_dict["lister"]["contacts"]["inquiry"][phone_field] = anonymize_phone(inquiry[phone_field])[0]
            
            # Anonymize viewing contact
            if "viewing" in contacts and isinstance(contacts["viewing"], dict):
                viewing = contacts["viewing"]
                
                # Anonymize email
                if "email" in viewing and viewing["email"]:
                    anon_dict["lister"]["contacts"]["viewing"]["email"] = anonymize_email(viewing["email"])[0]
                
                # Anonymize phones
                for phone_field in ["phone", "mobile"]:
                    if phone_field in viewing and viewing[phone_field]:
                        anon_dict["lister"]["contacts"]["viewing"][phone_field] = anonymize_phone(viewing[phone_field])[0]
    
    # Anonymize property address
    if "address" in anon_dict and isinstance(anon_dict["address"], dict):
        addr = anon_dict["address"]
        if addr.get("street") or addr.get("postalCode") or addr.get("locality") or addr.get("country"):
            full_hash, country_hash, zip_hash, city_hash, street_hash = anonymize_address(
                addr.get("street", ""),
                addr.get("postalCode", ""),
                addr.get("locality", ""),
                addr.get("country", "")
            )
            anon_dict["address"]["street"] = street_hash
            anon_dict["address"]["postalCode"] = zip_hash
            anon_dict["address"]["locality"] = city_hash
            anon_dict["address"]["country"] = country_hash
    
    return anon_dict


# Batch anonymization for Polars DataFrames
def anonymize_dataframe_batch(df: pl.DataFrame) -> pl.DataFrame:
    """
    Anonymizes PII columns in a Polars DataFrame in batch, including embedded PII in listing_json.
    
    This function processes the raw insertions DataFrame and replaces PII columns
    with their hashed equivalents. It also parses and anonymizes the listing_json field
    to remove all embedded PII before saving to disk.
    
    Args:
        df: Polars DataFrame with raw insertion data
        
    Returns:
        DataFrame with PII columns replaced by hashes and listing_json anonymized
    """
    import polars as pl
    import json
    
    df_anon = df.clone()
    
    # Anonymize user_ip_address using efficient map_elements
    if "user_ip_address" in df_anon.columns:
        def anonymize_ip_wrapper(ip: str) -> tuple:
            """Wrapper to handle None values."""
            if ip is None:
                return (None, None, None)
            return anonymize_ip(ip)
        
        # Get all IP hashes at once
        ip_hashes = df_anon["user_ip_address"].to_list()
        ip_results = [anonymize_ip_wrapper(ip) for ip in ip_hashes]
        
        df_anon = df_anon.with_columns([
            pl.Series("user_ip_address_hash", [r[0] for r in ip_results], dtype=pl.Utf8),
            pl.Series("user_ip_network_hash", [r[1] for r in ip_results], dtype=pl.Utf8),
            pl.Series("user_ip_host_hash", [r[2] for r in ip_results], dtype=pl.Utf8),
        ])
        df_anon = df_anon.drop("user_ip_address")  # Remove original PII
    
    # Anonymize contact_emails (comma-separated list)
    if "contact_emails" in df_anon.columns:
        def anonymize_email_list(email_str: str) -> str:
            """Anonymize comma-separated email list."""
            if not email_str:
                return None
            emails = [e.strip() for e in email_str.split(",") if e and e.strip()]
            hashes = [anonymize_email(e)[0] for e in emails if anonymize_email(e)[0]]
            return ",".join(hashes) if hashes else None
        
        email_hashes = df_anon["contact_emails"].to_list()
        email_results = [anonymize_email_list(emails) for emails in email_hashes]
        
        df_anon = df_anon.with_columns([
            pl.Series("contact_emails_hash", email_results, dtype=pl.Utf8)
        ])
        df_anon = df_anon.drop("contact_emails")  # Remove original PII
    
    # Anonymize listing_json - parse, anonymize, and re-serialize
    if "listing_json" in df_anon.columns:
        def anonymize_json_string(json_str: str) -> str:
            """Parse JSON, anonymize PII, and re-serialize."""
            if not json_str:
                return json_str
            
            try:
                listing_dict = json.loads(json_str)
                anon_dict = anonymize_listing_json_dict(listing_dict)
                return json.dumps(anon_dict, ensure_ascii=False)
            except (json.JSONDecodeError, TypeError, AttributeError) as e:
                # If JSON parsing fails, return original (log warning in production)
                print(f"Warning: Failed to parse/anonymize listing_json: {e}")
                return json_str
        
        # Process all listing_json strings
        json_strings = df_anon["listing_json"].to_list()
        anon_json_strings = [anonymize_json_string(js) for js in json_strings]
        
        df_anon = df_anon.with_columns([
            pl.Series("listing_json", anon_json_strings, dtype=pl.Utf8)
        ])
    
    return df_anon


def anonymize_listings_pii(df: "pl.DataFrame") -> "pl.DataFrame":
    """
    Anonymizes all PII fields extracted from listing_json during processing.
    
    This function should be called after process_listings() extracts fields from JSON.
    It anonymizes emails, phones, addresses, IPs, and names.
    
    Args:
        df: Polars DataFrame with extracted listing fields (from process_listings)
        
    Returns:
        DataFrame with PII fields replaced by hashes
    """
    import polars as pl
    
    df_anon = df.clone()
    
    # Anonymize email columns (using dot-notation field names)
    # Store full hash + component hashes (user_part, domain_part) for graph features
    email_cols = [
        "listing.lister.email",  # lister_email
        "listing.lister.billing.email",  # billing_email
        "listing.lister.contacts.inquiry.email",  # contact_inquiry_email
        "listing.lister.contacts.viewing.email"  # contact_viewing_email
    ]
    for col in email_cols:
        if col in df_anon.columns:
            emails = df_anon[col].to_list()
            email_results = [anonymize_email(e) if e else (None, None, None) for e in emails]
            # Keep dot notation in hash column names for consistency
            df_anon = df_anon.with_columns([
                pl.Series(f"{col}.hash", [r[0] for r in email_results], dtype=pl.Utf8),
                pl.Series(f"{col}.user_hash", [r[1] for r in email_results], dtype=pl.Utf8),
                pl.Series(f"{col}.domain_hash", [r[2] for r in email_results], dtype=pl.Utf8),
            ])
            df_anon = df_anon.drop(col)
    
    # Anonymize phone columns (using dot-notation field names)
    # Store full hash + component hashes (country, area, number) for graph features
    phone_cols = [
        "listing.lister.phone",  # lister_phone
        "listing.lister.mobile",  # lister_mobile (if exists)
        "listing.lister.billing.phoneDay",  # billing_phone_day
        "listing.lister.billing.phoneMobile",  # billing_phone_mobile
        "listing.lister.contacts.inquiry.phone",  # contact_inquiry_phone
        "listing.lister.contacts.inquiry.mobile",  # contact_inquiry_mobile
        "listing.lister.contacts.viewing.phone",  # contact_viewing_phone
        "listing.lister.contacts.viewing.mobile"  # contact_viewing_mobile
    ]
    for col in phone_cols:
        if col in df_anon.columns:
            phones = df_anon[col].to_list()
            phone_results = [anonymize_phone(p) if p else (None, None, None, None) for p in phones]
            # Keep dot notation in hash column names for consistency
            df_anon = df_anon.with_columns([
                pl.Series(f"{col}.hash", [r[0] for r in phone_results], dtype=pl.Utf8),
                pl.Series(f"{col}.country_hash", [r[1] for r in phone_results], dtype=pl.Utf8),
                pl.Series(f"{col}.area_hash", [r[2] for r in phone_results], dtype=pl.Utf8),
                pl.Series(f"{col}.number_hash", [r[3] for r in phone_results], dtype=pl.Utf8),
            ])
            df_anon = df_anon.drop(col)
    
    # Anonymize address components (using dot-notation field names)
    # We hash individual components to preserve graph structure while anonymizing
    # The address_id will be created from hashed components in create_nodes_and_edges
    address_groups = [
        # Note: lister address might not be in the flattened schema, skip if not present
        (["listing.lister.address.country", "listing.lister.address.postalCode", 
          "listing.lister.address.locality", "listing.lister.address.street"], "lister"),
        (["listing.lister.billing.address.country", "listing.lister.billing.address.postalCode",
          "listing.lister.billing.address.locality", "listing.lister.billing.address.street"], "billing"),
        (["listing.address.country", "listing.address.postalCode",
          "listing.address.locality", "listing.address.street"], "property")
    ]
    
    for cols, prefix in address_groups:
        if all(col in df_anon.columns for col in cols):
            # Get all address components
            country_col, zip_col, city_col, street_col = cols
            countries = df_anon[country_col].fill_null("").to_list()
            zips = df_anon[zip_col].fill_null("").to_list()
            cities = df_anon[city_col].fill_null("").to_list()
            streets = df_anon[street_col].fill_null("").to_list()
            
            # Anonymize each address
            address_hashes = [
                anonymize_address(street, zip_code, city, country)
                for street, zip_code, city, country in zip(streets, zips, cities, countries)
            ]
            
            # Add hashed components (keep dot notation for consistency)
            # For address groups, use the first column's path as base
            base_path = cols[0].rsplit(".", 1)[0] if "." in cols[0] else prefix
            df_anon = df_anon.with_columns([
                pl.Series(f"{base_path}.address_hash", [h[0] for h in address_hashes], dtype=pl.Utf8),
                pl.Series(f"{base_path}.country_hash", [h[1] for h in address_hashes], dtype=pl.Utf8),
                pl.Series(f"{base_path}.zip_hash", [h[2] for h in address_hashes], dtype=pl.Utf8),
                pl.Series(f"{base_path}.city_hash", [h[3] for h in address_hashes], dtype=pl.Utf8),
                pl.Series(f"{base_path}.street_hash", [h[4] for h in address_hashes], dtype=pl.Utf8),
            ])
            # Drop original address columns
            df_anon = df_anon.drop(cols)
    
    # Anonymize person names (using dot-notation field names)
    if "listing.lister.contacts.inquiry.givenName" in df_anon.columns and "listing.lister.contacts.inquiry.familyName" in df_anon.columns:
        given_names = df_anon["listing.lister.contacts.inquiry.givenName"].to_list()
        family_names = df_anon["listing.lister.contacts.inquiry.familyName"].to_list()
        name_hashes = [
            anonymize_person_name(g, f)[0] 
            if (g or f) else None
            for g, f in zip(given_names, family_names)
        ]
        df_anon = df_anon.with_columns([
            pl.Series("listing.lister.contacts.inquiry.name_hash", name_hashes, dtype=pl.Utf8)
        ])
        df_anon = df_anon.drop(["listing.lister.contacts.inquiry.givenName", "listing.lister.contacts.inquiry.familyName"])
    
    # Handle user_ip_address and contact_emails (should already be hashed, but check)
    if "user_ip_address" in df_anon.columns and "user_ip_address_hash" not in df_anon.columns:
        # Fallback: anonymize if not already done
        ips = df_anon["user_ip_address"].to_list()
        ip_hashes = [anonymize_ip(ip)[0] if ip else None for ip in ips]
        df_anon = df_anon.with_columns([
            pl.Series("user_ip_address_hash", ip_hashes, dtype=pl.Utf8)
        ])
        df_anon = df_anon.drop("user_ip_address")
    
    if "contact_emails" in df_anon.columns and "contact_emails_hash" not in df_anon.columns:
        # Fallback: anonymize if not already done
        def anonymize_email_list(email_str: str) -> str:
            if not email_str:
                return None
            emails = [e.strip() for e in email_str.split(",") if e and e.strip()]
            hashes = [anonymize_email(e)[0] for e in emails if anonymize_email(e)[0]]
            return ",".join(hashes) if hashes else None
        
        email_lists = df_anon["contact_emails"].to_list()
        email_hashes = [anonymize_email_list(emails) for emails in email_lists]
        df_anon = df_anon.with_columns([
            pl.Series("contact_emails_hash", email_hashes, dtype=pl.Utf8)
        ])
        df_anon = df_anon.drop("contact_emails")
    
    return df_anon

