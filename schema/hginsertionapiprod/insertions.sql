create table hginsertionapiprod.insertions
(
    id                      serial
        primary key,
    insertion_id            text    not null,
    object_reference        text    not null,
    listing                 jsonb
        constraint insertions_listing_lister_username_not_null
            check ((((listing -> 'lister'::text) ->> 'username'::text) IS NOT NULL) AND
                   (((listing -> 'lister'::text) ->> 'username'::text) <> ''::text)),
    selected_bundle         jsonb,
    status                  text,
    agency_id               text,
    created_at              timestamp with time zone,
    updated_at              timestamp with time zone,
    fakecheck_flag          boolean default false,
    last_migrated_at        timestamp with time zone,
    user_id                 integer not null
        references hginsertionapiprod.users
            on delete cascade,
    auto_approval_criteria  jsonb,
    fraud_flag              timestamp with time zone,
    version                 integer,
    last_published_version  integer,
    rejection_message       text,
    first_published_date    timestamp with time zone,
    contact_request_date    timestamp with time zone,
    is_contact_email_reused boolean,
    user_ip_address         text,
    note                    text,
    customer_segment        text,
    furthest_page           text,
    appointment             jsonb,
    listing_urls            jsonb,
    archival_reason         jsonb,
    should_hide_address     boolean,
    toplisting_data         jsonb,
    platform                text    not null,
    meta                    jsonb,
    listing_links           jsonb,
    constraint status_listing_id_not_null
        check (
            CASE
                WHEN (status = ANY (ARRAY ['PUBLISHED'::text, 'ARCHIVING'::text, 'REPUBLISHING'::text])) THEN (
                    ((listing -> 'id'::text) IS NOT NULL) AND ((listing ->> 'id'::text) <> ''::text))
                ELSE NULL::boolean
                END)
);

alter table hginsertionapiprod.insertions
    owner to postgres;

create unique index insertions_insertion_id_idx
    on hginsertionapiprod.insertions (insertion_id);

create index insertions_status_idx
    on hginsertionapiprod.insertions (status);

create index insertions_updated_at_idx
    on hginsertionapiprod.insertions (updated_at);

create index insertions_agency_id_idx
    on hginsertionapiprod.insertions (agency_id);

create index insertions_object_reference_idx_trgm
    on hginsertionapiprod.insertions using gin (object_reference gin_trgm_ops);

create index insertions_user_id_idx
    on hginsertionapiprod.insertions (user_id);

create unique index insertions_listing_id_idx
    on hginsertionapiprod.insertions ((listing ->> 'id'::text))
    where ((listing ->> 'id'::text) <> ''::text);

create index insertions_listing_address_street_idx
    on hginsertionapiprod.insertions (((listing -> 'address'::text) ->> 'street'::text));

create index insertions_listing_address_locality_idx
    on hginsertionapiprod.insertions (((listing -> 'address'::text) ->> 'locality'::text));

create index insertions_listing_address_postalcode_idx
    on hginsertionapiprod.insertions (((listing -> 'address'::text) ->> 'postalCode'::text));

create index insertions_listing_externalids_refobject_idx
    on hginsertionapiprod.insertions (((listing -> 'externalIds'::text) ->> 'refObject'::text));

create index insertions_listing_externalids_refhouse_idx
    on hginsertionapiprod.insertions (((listing -> 'externalIds'::text) ->> 'refHouse'::text));

create index insertions_listing_externalids_refproperty_idx
    on hginsertionapiprod.insertions (((listing -> 'externalIds'::text) ->> 'refProperty'::text));

create index insertions_listing_lister_username_idx
    on hginsertionapiprod.insertions (((listing -> 'lister'::text) ->> 'username'::text));

create index insertions_toplisting_listingtype_idx
    on hginsertionapiprod.insertions ((toplisting_data ->> 'listingType'::text));

create index insertions_from_crm_toplisting_listingtype_idx
    on hginsertionapiprod.insertions ((toplisting_data ->> 'listingType'::text));

create unique index insertions_object_reference_platform_idx
    on hginsertionapiprod.insertions (object_reference, platform);

create index insertions_object_reference_idx
    on hginsertionapiprod.insertions (object_reference);

grant select on hginsertionapiprod.insertions to hginsertionapiprod_readonly_user;

