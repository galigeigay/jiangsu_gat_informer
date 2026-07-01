CREATE TABLE ads_terraqt_station_realtime (
    station_code VARCHAR(10),
    date_time DATETIME,
    update_time DATETIME,
    province VARCHAR(255) DEFAULT NULL,
    city VARCHAR(255) DEFAULT NULL,
    district VARCHAR(255) DEFAULT NULL,
    pressure DOUBLE DEFAULT NULL,
    pressure_at_sea DOUBLE DEFAULT NULL,
    windspeed_max DOUBLE DEFAULT NULL,
    windspeed_inst_max DOUBLE DEFAULT NULL,
    winddir_inst_max DOUBLE DEFAULT NULL,
    winddir_avg_2min DOUBLE DEFAULT NULL,
    winddir_max DOUBLE DEFAULT NULL,
    temperature DOUBLE DEFAULT NULL,
    humidity DOUBLE DEFAULT NULL,
    precipitation_1h DOUBLE DEFAULT NULL COMMENT '1小时降水量 单位mm',
    visibility DOUBLE DEFAULT NULL
) ENGINE = OLAP
UNIQUE KEY (station_code, date_time)
PARTITION BY RANGE(date_time)
(
    PARTITION p1 VALUES LESS THAN ('2022-01-01 00:00:00'),
    PARTITION p2 VALUES LESS THAN ('2023-01-01 00:00:00'),
    PARTITION p3 VALUES LESS THAN ('2024-01-01 00:00:00'),
    PARTITION p4 VALUES LESS THAN MAXVALUE
)
DISTRIBUTED BY HASH(date_time)
PROPERTIES (
    "replication_allocation" = "tag.location.default: 1"
);