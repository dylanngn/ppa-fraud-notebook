create table hginsertionapiprod.status_history
(
    id                   serial
        primary key,
    status_from          text    not null,
    status_to            text    not null,
    transition_timestamp timestamp with time zone,
    username             text,
    insertion_id         integer not null
        references hginsertionapiprod.insertions
            on delete cascade
);

alter table hginsertionapiprod.status_history
    owner to postgres;

create unique index status_history_unique_idx
    on hginsertionapiprod.status_history (status_from, status_to, transition_timestamp, insertion_id);

create index status_history_insertion_id_idx
    on hginsertionapiprod.status_history (insertion_id);

grant select on hginsertionapiprod.status_history to hginsertionapiprod_readonly_user;

