#!/usr/bin/env bash

db2 "connect to se3db3"
db2 -x "select 'drop table ' || rtrim(tabschema) || '.' || rtrim(tabname) || ';' from syscat.tables where tabschema='PHAMA10'"
db2 "connect reset"
