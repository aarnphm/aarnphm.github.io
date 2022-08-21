connect to SE3DB3;

------------------------------------------------
--  DDL Statement for table PERSON
------------------------------------------------
CREATE TABLE PERSON (
    ID INTEGER NOT NULL,
    NAME VARCHAR(100) NOT NULL,
    AGE SMALLINT,
    PHONE_NUMBER VARCHAR(20),
    PRIMARY KEY (ID)
);

------------------------------------------------
--  DDL Statement for table COUNTRY
------------------------------------------------
CREATE TABLE COUNTRY (
    CODE CHAR(3) NOT NULL,
    NAME VARCHAR(100) NOT NULL,
    CONTINENT VARCHAR(50) NOT NULL,
    PRIMARY KEY (CODE)
);

------------------------------------------------
--  DDL Statement for table AIRLINE
------------------------------------------------
CREATE TABLE AIRLINE (
    ALIAS CHAR(2) NOT NULL,
    NAME VARCHAR(100) NOT NULL,
    COUNTRY_CODE CHAR(3) NOT NULL,
    PRIMARY KEY (ALIAS),
    FOREIGN KEY (COUNTRY_CODE) REFERENCES COUNTRY(CODE)
);

------------------------------------------------
--  DDL Statement for table AIRPLANE
------------------------------------------------
CREATE TABLE AIRPLANE (
    SERIAL_NUMBER VARCHAR(50) NOT NULL,
    MANUFACTURER VARCHAR(100) NOT NULL,
    MODEL VARCHAR(50) NOT NULL,
    AIRLINE_ALIAS CHAR(2) NOT NULL,
    PRIMARY KEY (SERIAL_NUMBER),
    FOREIGN KEY (AIRLINE_ALIAS) REFERENCES AIRLINE(ALIAS)
);

------------------------------------------------
--  DDL Statement for table AIRPORT
------------------------------------------------
CREATE TABLE AIRPORT (
    IATA_CODE CHAR(3) NOT NULL,
    NAME VARCHAR(100) NOT NULL,
    CITY VARCHAR(100) NOT NULL,
    COUNTRY_CODE CHAR(3) NOT NULL,
    PRIMARY KEY (IATA_CODE),
    FOREIGN KEY (COUNTRY_CODE) REFERENCES COUNTRY(CODE)
);

------------------------------------------------
--  DDL Statement for table PASSENGER
------------------------------------------------
CREATE TABLE PASSENGER (
    PERSON_ID INTEGER NOT NULL,
    DIETARY_PREFERENCE VARCHAR(50),
    PRIMARY KEY (PERSON_ID),
    FOREIGN KEY (PERSON_ID) REFERENCES PERSON(ID)
);

------------------------------------------------
--  DDL Statement for table PILOT
------------------------------------------------
CREATE TABLE PILOT (
    PERSON_ID INTEGER NOT NULL,
    POSITION VARCHAR(50) NOT NULL,
    SALARY DECIMAL(10,2) NOT NULL,
    PRIMARY KEY (PERSON_ID),
    FOREIGN KEY (PERSON_ID) REFERENCES PERSON(ID)
);

------------------------------------------------
--  DDL Statement for table CABIN_CREW
------------------------------------------------
CREATE TABLE CABIN_CREW (
    PERSON_ID INTEGER NOT NULL,
    POSITION VARCHAR(50) NOT NULL,
    SALARY DECIMAL(10,2) NOT NULL,
    AIRLINE_ALIAS CHAR(2),
    PRIMARY KEY (PERSON_ID),
    FOREIGN KEY (PERSON_ID) REFERENCES PERSON(ID),
    FOREIGN KEY (AIRLINE_ALIAS) REFERENCES AIRLINE(ALIAS)
);

------------------------------------------------
--  DDL Statement for table GROUND_STAFF
------------------------------------------------
CREATE TABLE GROUND_STAFF (
    PERSON_ID INTEGER NOT NULL,
    SALARY DECIMAL(10,2) NOT NULL,
    DEPARTMENT VARCHAR(50) NOT NULL,
    AIRPORT_CODE CHAR(3),
    PRIMARY KEY (PERSON_ID),
    FOREIGN KEY (PERSON_ID) REFERENCES PERSON(ID),
    FOREIGN KEY (AIRPORT_CODE) REFERENCES AIRPORT(IATA_CODE)
);

------------------------------------------------
--  DDL Statement for table ROUTE
------------------------------------------------
CREATE TABLE ROUTE (
    ID INTEGER NOT NULL,
    NUM_STOPS SMALLINT NOT NULL,
    DURATION DECIMAL(5,2) NOT NULL,
    SOURCE_AIRPORT CHAR(3) NOT NULL,
    DESTINATION_AIRPORT CHAR(3) NOT NULL,
    PRIMARY KEY (ID),
    FOREIGN KEY (SOURCE_AIRPORT) REFERENCES AIRPORT(IATA_CODE),
    FOREIGN KEY (DESTINATION_AIRPORT) REFERENCES AIRPORT(IATA_CODE)
);

------------------------------------------------
--  DDL Statement for table SCHEDULED_FLIGHT
------------------------------------------------
CREATE TABLE SCHEDULED_FLIGHT (
    FLIGHT_NUMBER VARCHAR(10) NOT NULL,
    DEPARTURE_DATE DATE NOT NULL,
    ARRIVAL_DATE DATE NOT NULL,
    SCHEDULED_DEPARTURE_TIME TIME NOT NULL,
    SCHEDULED_ARRIVAL_TIME TIME NOT NULL,
    ACTUAL_DEPARTURE_TIME TIME,
    ACTUAL_ARRIVAL_TIME TIME,
    AIRLINE_ALIAS CHAR(2) NOT NULL,
    ROUTE_ID INTEGER NOT NULL,
    PRIMARY KEY (FLIGHT_NUMBER, DEPARTURE_DATE),
    FOREIGN KEY (AIRLINE_ALIAS) REFERENCES AIRLINE(ALIAS),
    FOREIGN KEY (ROUTE_ID) REFERENCES ROUTE(ID)
);

------------------------------------------------
--  DDL Statement for table TICKET
------------------------------------------------
CREATE TABLE TICKET (
    TICKET_NUMBER CHAR(13) NOT NULL,
    SEAT_NUMBER VARCHAR(4) NOT NULL,
    CLASS CHAR(1) NOT NULL,
    PRICE DECIMAL(10,2) NOT NULL,
    TRAVEL_WEBSITE VARCHAR(50) NOT NULL,
    PASSENGER_ID INTEGER NOT NULL,
    FLIGHT_NUMBER VARCHAR(10) NOT NULL,
    DEPARTURE_DATE DATE NOT NULL,
    PRIMARY KEY (TICKET_NUMBER),
    FOREIGN KEY (PASSENGER_ID) REFERENCES PASSENGER(PERSON_ID),
    FOREIGN KEY (FLIGHT_NUMBER, DEPARTURE_DATE) REFERENCES SCHEDULED_FLIGHT(FLIGHT_NUMBER, DEPARTURE_DATE),
    CHECK (CLASS IN ('E', 'B', 'F'))
);

------------------------------------------------
--  DDL Statement for table BAGGAGE
------------------------------------------------
CREATE TABLE BAGGAGE (
    ID INTEGER NOT NULL,
    TICKET_NUMBER CHAR(13) NOT NULL,
    TYPE VARCHAR(20) NOT NULL,
    QUANTITY SMALLINT NOT NULL,
    TOTAL_WEIGHT DECIMAL(5,2) NOT NULL,
    IS_FRAGILE BOOLEAN NOT NULL,
    PRIMARY KEY (ID),
    FOREIGN KEY (TICKET_NUMBER) REFERENCES TICKET(TICKET_NUMBER)
);

------------------------------------------------
--  DDL Statement for table PILOT_AIRPLANE
------------------------------------------------
CREATE TABLE PILOT_AIRPLANE (
    PILOT_ID INTEGER NOT NULL,
    AIRPLANE_SERIAL_NUMBER VARCHAR(50) NOT NULL,
    PRIMARY KEY (PILOT_ID, AIRPLANE_SERIAL_NUMBER),
    FOREIGN KEY (PILOT_ID) REFERENCES PILOT(PERSON_ID),
    FOREIGN KEY (AIRPLANE_SERIAL_NUMBER) REFERENCES AIRPLANE(SERIAL_NUMBER)
);

------------------------------------------------
--  DDL Statement for table AIRLINE_ROUTE
------------------------------------------------
CREATE TABLE AIRLINE_ROUTE (
    AIRLINE_ALIAS CHAR(2) NOT NULL,
    ROUTE_ID INTEGER NOT NULL,
    PRIMARY KEY (AIRLINE_ALIAS, ROUTE_ID),
    FOREIGN KEY (AIRLINE_ALIAS) REFERENCES AIRLINE(ALIAS),
    FOREIGN KEY (ROUTE_ID) REFERENCES ROUTE(ID)
);

connect reset;
