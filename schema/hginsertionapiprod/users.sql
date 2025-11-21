create table hginsertionapiprod.users
(
    id                serial
        primary key,
    owner_id          text                                     not null,
    created_at        timestamp with time zone,
    updated_at        timestamp with time zone,
    contact_emails    text,
    source            text    default 'INSERTION_FUNNEL'::text not null,
    platform          text                                     not null,
    user_type         text    default 'PRIVATE'::text,
    auto_segmentation boolean default true,
    deleted_at        timestamp with time zone
);

alter table hginsertionapiprod.users
    owner to postgres;

create unique index users_owner_id_platform_idx
    on hginsertionapiprod.users (owner_id, platform);

grant select on hginsertionapiprod.users to hginsertionapiprod_readonly_user;

