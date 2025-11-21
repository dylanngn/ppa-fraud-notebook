create type hginsertionapiprod.payment_subscription_status as enum ('DRAFT', 'ACTIVE', 'CANCELED');

alter type hginsertionapiprod.payment_subscription_status owner to postgres;

