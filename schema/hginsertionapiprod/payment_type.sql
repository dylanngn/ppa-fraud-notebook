create type hginsertionapiprod.payment_type as enum ('INVOICE', 'DIRECT');

alter type hginsertionapiprod.payment_type owner to postgres;

