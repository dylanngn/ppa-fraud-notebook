# Knowledge Base: Fraud Detection Feature Engineering

This document tracks our understanding of the data, feature definitions, and engineering logic.

## 1. Data Sources & Identifiers

*   **Insertion ID**: `object_reference` (Text) is the primary business identifier for a listing insertion. `id` (Integer) is the internal DB key.
*   **User ID**: `owner_id` (Text) identifies the user across the system.
*   **Platform**: `platform` (Text) indicates the brand (Homegate, ImmoScout24, SMG).

## 2. Fraud Logic & Labels

*   **Fraud Flag**: `fraud_flag` (Timestamp). If present, the listing was marked as fraud.
*   **Seon (Baseline)**: `auto_approval_criteria` contains the result from the existing Seon fraud check.
*   **Slip-Through Fraud**:
    *   If `fraud_flag` > `first_published_date`, the listing was initially approved (passed Seon) but later caught as fraud. This is the **critical** target for our model. We want to catch these *before* they are published.

## 3. Feature Extraction (Listing JSON)

We need to extract the following deep features from the `listing` JSONB:

*   **Lister Info**:
    *   `lister.username`: Account username.
    *   `lister.email`: Contact email? (Need to verify schema).
    *   `lister.address`: Billing address?
*   **Contact Info**:
    *   `contactForm`?
*   **Billing Info**:
    *   (Need to verify exact path in schema).

## 4. Temporal Filtering

*   **Submission Date**: The timestamp when status changed from `DRAFT` to `PENDING_APPROVAL`.
*   **Scope**: Listings submitted between **2023-11-01** and **2025-11-01**.

## 5. Graph Nodes & Edges

*   **User Node**:
    *   ID: `owner_id`
    *   Features: `user_type`, `email_domain` (from listing contact/billing).
*   **Listing Node**:
    *   ID: `object_reference`
    *   Features: `price`, `size`, `rooms`, `zip`, `city`, `platform`, `auto_approval_criteria` (as feature?), `embeddings`.
*   **IP Node**:
    *   ID: `user_ip_address`
*   **Email Node**:
    *   ID: Email Address (normalized).
    *   Sources: 
        *   `users.contact_emails`
        *   `listing.lister.email`
        *   `listing.lister.billing.email`
        *   `listing.lister.contacts.inquiry.email`
        *   `listing.lister.contacts.viewing.email`
*   **Phone Node**:
    *   ID: Phone Number (normalized).
    *   Sources:
        *   `listing.lister.phone`, `listing.lister.mobile`
        *   `listing.lister.billing.phoneDay`, `listing.lister.billing.phoneEvening`, `listing.lister.billing.phoneMobile`
        *   `listing.lister.contacts.inquiry.phone`, `listing.lister.contacts.inquiry.mobile`
        *   `listing.lister.contacts.viewing.phone`, `listing.lister.contacts.viewing.mobile`
*   **Location Node** (New):
    *   ID: `ZipCode_City` (Composite Key).
    *   Sources:
        *   Property Address: `listing.address`
        *   Lister Address: `listing.lister.address`
        *   Billing Address: `listing.lister.billing.address`
*   **Edges**:
    *   User -> Posts -> Listing
    *   User -> Uses -> IP
    *   User -> Has -> Email
    *   Listing -> Has -> Email
    *   Listing -> Has -> Phone
    *   Listing -> Located_At -> Location
    *   User -> Located_At -> Location (via Lister/Billing address)
