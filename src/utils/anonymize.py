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

