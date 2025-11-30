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
*   **Localization**:
    *   `localization`: Contains localized text (title, description) for `de`, `en`, `fr`, `it`.
    *   `primary`: The declared primary language of the listing.
    *   **Logic**: We extract the `primary` language code and coalesce the description text (prioritizing primary, then falling back to others) for embedding.
*   **Customer Segment**:
    *   `customer_segment`: Self-identified role (Tenant, Owner, Business).
*   **Bundle Info**:
    *   `selected_bundle`: Extracted `period` (duration) and `tier` (Basic, Premium, Top).
*   **Payment Info**:
    *   `lister.billing.payment.paymentType`: Method of payment (DIRECT vs INVOICE).

## 4. Temporal Filtering

*   **Submission Date**: The timestamp when status changed from `DRAFT` to `PENDING_APPROVAL`.
*   **Scope**: Listings submitted between **2023-01-01** and **2025-11-02**.
*   **Evaluation Strategy** (Sliding Window):
    *   **Training Window** (`window_days`): **90 days** (default). The model learns from this historical period.
    *   **Step Size** (`step_days`): **14 days** (default). How far the window moves forward for the next iteration.
    *   **Test Window** (`test_size`): **14 days**. The future period evaluated after training.
    *   **Why Step Size?**: It determines the *frequency* of retraining. Setting `Step Size = Test Window` ensures contiguous, non-overlapping evaluation (we predict every day exactly once).

## 5. Graph Nodes & Edges

*   **User Node**:
    *   ID: `owner_id`
    *   Features: `account_age`, `email_domain`.
*   **Listing Node**:
    *   ID: `object_reference`
    *   Features: 
        *   Core: `offer_type`, `price`, `living_space`, `rooms`, `zip`, `city`, `platform`.
        *   Enhanced: `bundle_tier` (Ordinal), `payment_type` (Binary), `customer_segment` (One-Hot), `language` (One-Hot).
        *   Embeddings: Description text embedding (multilingual).
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
    *   ID: `Country_Zip_City_Street` (Composite Key).
    *   Features: `latitude`, `longitude`.
    *   **Types/Edges**:
        *   `LOCATED_AT` (Property Address: `listing.address`) - The bait.
        *   `LISTER_ADDRESS` (Lister Address: `lister.address`) - The declared entity.
        *   `BILLING_ADDRESS` (Billing Address: `lister.billing.address`) - The money trail.