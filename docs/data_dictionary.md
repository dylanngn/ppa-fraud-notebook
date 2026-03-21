# Data Dictionary & Retraining Requirements

Because the original raw dataset contains sensitive PII and is proprietary to Swiss Marketplace Group (SMG), it is not included in this repository. To retrain the models or validate the pipeline on new data, you must provide two CSV files that match the expected schema.

## 1. Events Dataset (`insertion_events.csv`)

This dataset captures the lifecycle events of real estate listings (e.g., transitions from DRAFT to PUBLISHED or DELETED).

### Required System Columns
The ETL pipeline (`src/data/extract.py`) strictly relies on the following columns for data joining, temporal splitting, and label derivation:

| Column Name | Expected Format | Description |
|-------------|-----------------|-------------|
| `OBJECTREFERENCE` | String containing regex pattern `#<ID>##` | The unique insertion ID is extracted from this string using the regex pattern `#(.+?)##`. Example: `Listings#A123BC##` extracts to `A123BC`. |
| `DATAPIPELINE_EVENT_SENT_AT` | String `YYYY-MM-DD HH:MM:SS.f +ZZZZ` | The event timestamp. Example: `2025-01-15 10:30:45.123 +0100`. Must be parseable to infer temporal point-in-time state. |
| `STATUS` | String (Categorical) | The listing status at the time of the event (e.g., `DRAFT`, `PUBLISHED`, `DELETED`). |
| `FLAGGEDFORFRAUD` | String (Timestamp) or Null | The exact timestamp when a listing was flagged as fraud. A listing is considered fraudulent if this column is not null. *Note: this timestamp is used to construct point-in-time labels.* |

### Required Feature Columns
In addition, the following categorical, numeric, and PII columns are expected to build the features and heterogeneous graph:
- **PII / Identifiers:** `LISTING_LISTER_EMAIL`, `LISTING_LISTER_PHONE`, `user_id`, `ip`, `session/device_hash`. (These will be automatically SHA-256 anonymized during ETL).
- **Listing Attributes:** `LISTING_PRICES_RENT_NET`, `LISTING_PRICES_AMOUNT`, `LISTING_ADDRESS_LATITUDE`, `LISTING_ADDRESS_LONGITUDE`, `LISTING_ADDRESS_COUNTRY`, `TARGETPLATFORM`, `LISTING_OFFERTYPE`, `LISTING_CATEGORIES`.
- **Billing Attributes:** `billing_country`, `payment_mode`, `action_type`.

---

## 2. SEON Transactions Dataset (`seon_transactions.csv`)

This dataset contains the third-party fraud detection API responses from SEON, capturing risk signals for each transaction/listing creation.

### Required System Columns
| Column Name | Expected Format | Description |
|-------------|-----------------|-------------|
| `transaction_id` | String | Must match the extracted `INSERTION_ID` from the events dataset to perform the as-of join. |
| `id` | String | A unique SEON system identifier for the api call. |
| `date` | String `YYYY-MM-DDTHH:MM:SS.f+ZZZZ` | Timestamp of the SEON API response. Example: `2025-01-15T10:30:45.123+0000`. Used for the temporal as-of join. |

### Required Feature Columns
Raw signals returned from the SEON API used to build the final hybrid model features:
- **Boolean Risk Signals:** `tor`, `vpn`, `data_center_proxy`, `residential_proxy`, `public_proxy`, `email/domain/disposable`, `phone_is_disposable`, `phone_is_valid`.
- **Categorical Session Data:** `ip_type`, `ip_country`, `ip_isp_name`, `session/os`, `session/browser`, `session/device_type`, `session/screen_resolution`, `session/adblock`, `phone_carrier`, `email/deliverable`, `email/domain_registered`, `email/dmarc_enforced`.

*(Note: Target prediction scores output by SEON directly such as `fraud_score`, `blackbox_score` or `email_score` are analyzed but strictly excluded from our model feature inputs to avoid target leakage.)*

---

## Retraining Workflow

Once your data is prepared:
1. Place both CSV files into the `artifacts/` directory as `insertion_events.csv` and `seon_transactions.csv`.
2. Run the ETL pipeline: `python -m src.data.etl`
3. Launch retraining (default Vanilla XGBoost): `python -m src.training.trainer`
