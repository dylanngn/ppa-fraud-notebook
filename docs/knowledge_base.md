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

*   **Offer Type**: `offerType` (BUY vs RENT).
    *   **Price**: Extract `prices.buy.price` OR `prices.rent.gross` (or `net` if gross missing).
*   **Characteristics**:
    *   Numeric: `livingSpace`, `numberOfRooms`, `yearBuilt`, `floor`, `numberOfFloors`.
    *   Boolean: `isNewBuilding`, `hasBalcony`, `hasElevator`, `hasParking`, `isOldBuilding`.
*   **Lister Info**:
    *   `lister.username`: Account username (often the registration email).
    *   `lister.email`: Contact email.
    *   `lister.phone`, `lister.mobile`: Contact phones.
    *   `lister.address`: Lister's physical address.
*   **Billing Info** (Sensitive/Private):
    *   `lister.billing.email`: Email for invoices (often different from contact).
    *   `lister.billing.address`: Address for invoices.
    *   `lister.billing.phoneDay`, `phoneMobile`: Billing phones.
*   **Contact Persons**:
    *   `lister.contacts.inquiry`: The person handling inquiries.
        *   `givenName`, `familyName`, `email`, `phone`, `mobile`.

## 4. Temporal Filtering

*   **Submission Date**: The timestamp when status changed from `DRAFT` to `PENDING_APPROVAL`.
*   **Scope**: Listings submitted between **2023-11-01** and **2025-11-01**.

## 5. Graph Nodes & Edges

*   **User Node**:
    *   ID: `owner_id`
    *   Features: `account_age`, `email_domain`.
*   **Listing Node**:
    *   ID: `object_reference`
    *   Features: `offer_type`, `price`, `living_space`, `rooms`, `zip`, `city`, `platform`, `characteristics_vector`.
*   **IP Node**:
    *   ID: `user_ip_address`
*   **Email Node**:
    *   ID: Email Address (normalized).
    *   **Types/Edges**:
        *   `HAS_CONTACT_EMAIL` (from `lister.email`, `contacts.inquiry.email`) - Visible to seekers.
        *   `HAS_BILLING_EMAIL` (from `lister.billing.email`) - Private, likely real owner.
        *   `HAS_USERNAME_EMAIL` (from `lister.username`) - Account login.
*   **Phone Node**:
    *   ID: Phone Number (normalized).
    *   **Types/Edges**:
        *   `HAS_CONTACT_PHONE` (from `lister.phone`, `contacts.inquiry.phone`)
        *   `HAS_BILLING_PHONE` (from `lister.billing.phone*`)
*   **Address Node** (New Granularity):
    *   ID: `Street_Zip_City` (Composite Key).
    *   **Types/Edges**:
        *   `LOCATED_AT` (Property Address: `listing.address`) - The bait.
        *   `LISTER_ADDRESS` (Lister Address: `lister.address`) - The declared entity.
        *   `BILLING_ADDRESS` (Billing Address: `lister.billing.address`) - The money trail.
*   **Person Node** (New):
    *   ID: `GivenName_FamilyName` (Normalized).
    *   Source: `lister.contacts.inquiry.givenName` + `familyName`.
    *   Edge: `Listing -> HAS_CONTACT_PERSON -> Person`.
