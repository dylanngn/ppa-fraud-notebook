create table hginsertionapiprod.payment_subscriptions
(
    id                                     serial
        primary key,
    payment_type                           hginsertionapiprod.payment_type                not null,
    zuora_subscription_number              varchar(255),
    converted_payment_type_to_invoice_date timestamp with time zone,
    object_reference                       varchar(50)                                    not null,
    platform                               varchar(50)                                    not null,
    status                                 hginsertionapiprod.payment_subscription_status not null,
    created_at                             timestamp with time zone default now()         not null,
    updated_at                             timestamp with time zone default now()         not null,
    deleted_at                             timestamp with time zone
);

alter table hginsertionapiprod.payment_subscriptions
    owner to postgres;

create unique index payment_subscriptions_object_reference_platform_idx
    on hginsertionapiprod.payment_subscriptions (object_reference, platform);

create index payment_subscriptions_status_idx
    on hginsertionapiprod.payment_subscriptions (status);

create index payment_subscriptions_payment_type_idx
    on hginsertionapiprod.payment_subscriptions (payment_type);

grant select on hginsertionapiprod.payment_subscriptions to hginsertionapiprod_readonly_user;

