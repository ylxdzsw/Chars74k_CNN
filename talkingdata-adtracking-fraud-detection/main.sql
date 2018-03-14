.open D:/talkingdata-adtracking-fraud-detection/clean/db.sqlite

-- config: we only care about the speed --
PRAGMA synchronous = OFF;
PRAGMA journal_mode = MEMORY; -- OFF?
PRAGMA locking_mode = EXCLUSIVE;
PRAGMA threads = 8;
PRAGMA mmap_size = 1073741824; -- 1GB

.timer on

-- import raw data --
.mode csv
.import D:/talkingdata-adtracking-fraud-detection/raw/train.csv raw_train
.import D:/talkingdata-adtracking-fraud-detection/raw/test.csv raw_test

-- the main table --
create table main(
    id integer primary key not null,
    ip integer not null,
    app integer not null,
    device integer not null,
    os integer not null,
    channel integer not null,
    clicktime integer not null,
    attrtime integer,
    clickid integer
);

create index app on main(app);
create index device on main(device);
create index os on main(os);
create index channel on main(channel);
create index clicktime on main(clicktime);

-- load training data --
insert into main(ip, app, device, os, channel, clicktime, attrtime)
select ip, app, device, os, channel,
       strftime('%s', click_time),
       strftime('%s', attributed_time)
from raw_train;

-- load testing data --
insert into main(ip, app, device, os, channel, clicktime, clickid)
select ip, app, device, os, channel,
       strftime('%s', click_time),
       click_id
from raw_test;

-- day, hour, minute, and second --
create table clicktimes(
    id integer primary key not null,
    day integer not null,
    hour integer not null,
    minute integer not null,
    second integer not null
);

insert into clicktimes
select id,
       strftime('%d', clicktime, 'unixepoch'),
       strftime('%H', clicktime, 'unixepoch'),
       strftime('%M', clicktime, 'unixepoch'),
       strftime('%S', clicktime, 'unixepoch')
from main;

