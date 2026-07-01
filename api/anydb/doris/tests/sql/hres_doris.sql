CREATE TABLE ads_terraqt_ecmwf_hres_by_station_v (
    station_code VARCHAR(10),
    date_time DATETIME,
    version DATETIME,
    province VARCHAR(255) DEFAULT NULL,
    city VARCHAR(255) DEFAULT NULL,
    district VARCHAR(255) DEFAULT NULL,
    coordinates VARCHAR(255),
    temperature_2m DOUBLE DEFAULT NULL,
    dewpoint_2m DOUBLE DEFAULT NULL,
    pressure_surface DOUBLE DEFAULT NULL,
    apparent_temperature DOUBLE DEFAULT NULL,
    humidity DOUBLE DEFAULT NULL,
    cloudcover DOUBLE DEFAULT NULL,
    cloudcover_low DOUBLE DEFAULT NULL,
    cloudcover_medium DOUBLE DEFAULT NULL,
    cloudcover_high DOUBLE DEFAULT NULL,
    windspeed_10m DOUBLE DEFAULT NULL,
    windspeed_100m DOUBLE DEFAULT NULL,
    windspeed_200m DOUBLE DEFAULT NULL,
    precipitation_instant DOUBLE DEFAULT NULL,
    shortwave_radiation_instant DOUBLE DEFAULT NULL,
    snow_fall_instant DOUBLE DEFAULT NULL,
    total_precipitation DOUBLE DEFAULT NULL,
    shortwave_radiation DOUBLE DEFAULT NULL,
    snow_fall DOUBLE DEFAULT NULL COMMENT '累计降雪量，单位米'
) ENGINE=OLAP
UNIQUE KEY (station_code, date_time, version)
PARTITION BY RANGE(version)
(
    PARTITION p1 VALUES LESS THAN ('2022-01-01 00:00:00'),
    PARTITION p2 VALUES LESS THAN ('2023-01-01 00:00:00'),
    PARTITION p3 VALUES LESS THAN ('2024-01-01 00:00:00'),
    PARTITION p4 VALUES LESS THAN MAXVALUE
)
DISTRIBUTED BY HASH(version)
PROPERTIES (
    "replication_allocation" = "tag.location.default: 1"
);