connect to se3db3;

--++++++++++++++++++++++++++++++++++++++++++++++
-- CREATE TABLES
--++++++++++++++++++++++++++++++++++++++++++++++

------------------------------------------------
--  DDL Statements for table Country
------------------------------------------------
CREATE TABLE Country (
    Code VARCHAR(10) NOT NULL PRIMARY KEY,
    Name VARCHAR(100) NOT NULL,
    Continent VARCHAR(50) NOT NULL
);


------------------------------------------------
--  DDL Statements for table Airline
------------------------------------------------
CREATE TABLE Airline (
    Alias VARCHAR(20) NOT NULL PRIMARY KEY,
    Name VARCHAR(100) NOT NULL,
    CountryCode VARCHAR(10) NOT NULL,
    FOREIGN KEY(CountryCode) references Country(Code) on delete cascade
);


------------------------------------------------
--  DDL Statements for table Airplane
------------------------------------------------
CREATE TABLE Airplane (
    SerialNo VARCHAR(50) NOT NULL PRIMARY KEY,
    Company VARCHAR(100) NOT NULL,
    Model VARCHAR(100) NOT NULL,
    AirlineAlias VARCHAR(50) NOT NULL,
    FOREIGN KEY(AirlineAlias) references Airline(Alias) on delete cascade
);


------------------------------------------------
--  DDL Statements for table Airport
------------------------------------------------
CREATE TABLE Airport (
    IATA VARCHAR(10) NOT NULL PRIMARY KEY,
    Name VARCHAR(100),
    City VARCHAR(100),
    CountryCode VARCHAR(10) NOT NULL,
    FOREIGN KEY(CountryCode) references Country(Code) on delete cascade
);

------------------------------------------------
--  DDL Statements for table Route
------------------------------------------------
CREATE TABLE Route (
    RouteID INT NOT NULL PRIMARY KEY,
    srcAirport VARCHAR(10) NOT NULL,
    dstAirport VARCHAR(10) NOT NULL,
    Stops INT,
    Duration INT,
    FOREIGN KEY(srcAirport) references Airport(IATA) on delete cascade,
    FOREIGN KEY(dstAirport) references Airport(IATA) on delete cascade
);

------------------------------------------------
--  DDL Statements for table ScheduledFlight
------------------------------------------------
CREATE TABLE ScheduledFlight (
    FlightNo VARCHAR(50) NOT NULL,
    ScheduledDepTime TIME,
    ActualDepTime TIME,
    ScheduledArrTime TIME,
    ActualArrTime TIME,
    DepDate DATE NOT NULL,
    ArrDate DATE NOT NULL,
    RouteID INT NOT NULL,
    AirlineAlias VARCHAR(20) NOT NULL,
    PRIMARY KEY(FlightNo, DepDate),
    FOREIGN KEY(AirlineAlias) references Airline(Alias) on delete cascade,
    FOREIGN KEY(RouteID) references Route(RouteID) on delete cascade
);

------------------------------------------------
--  DDL Statements for table Person
------------------------------------------------
CREATE TABLE Person (
    PersonID INT NOT NULL PRIMARY KEY,
    Name VARCHAR(100) NOT NULL,
    Age INT NOT NULL,
    PhoneNo VARCHAR(20)
);

------------------------------------------------
--  DDL Statements for table Pilot
------------------------------------------------
CREATE TABLE Pilot (
    PersonID INT NOT NULL PRIMARY KEY,
    Position VARCHAR(50),
    Salary DECIMAL(10,2),
    FOREIGN KEY (PersonID) REFERENCES Person(PersonID) on delete cascade
);

------------------------------------------------
--  DDL Statements for table CabinCrew
------------------------------------------------
CREATE TABLE CabinCrew (
    PersonID INT NOT NULL PRIMARY KEY,
    Position VARCHAR(50),
    Salary DECIMAL(10,2),
    AirlineAlias VARCHAR(50),
    FOREIGN KEY (PersonID) REFERENCES Person(PersonID) on delete cascade,
    FOREIGN KEY (AirlineAlias) REFERENCES Airline(Alias) on delete cascade
);

------------------------------------------------
--  DDL Statements for table GroundStaff
------------------------------------------------
CREATE TABLE GroundStaff (
    PersonID INT NOT NULL PRIMARY KEY,
    Department VARCHAR(100),
    Salary DECIMAL(10,2),
    AirportIATA VARCHAR(10),
    FOREIGN KEY (PersonID) REFERENCES Person(PersonID),
    FOREIGN KEY (AirportIATA) REFERENCES Airport(IATA)
);

------------------------------------------------
--  DDL Statements for table Passenger
------------------------------------------------
CREATE TABLE Passenger (
    PersonID INT NOT NULL PRIMARY KEY,
    DietaryPref VARCHAR(100),
    FOREIGN KEY (PersonID) REFERENCES Person(PersonID) on delete cascade
);

------------------------------------------------
--  DDL Statements for table Ticket
------------------------------------------------
CREATE TABLE Ticket (
    TicketNo VARCHAR(50) NOT NULL PRIMARY KEY,
    SeatNo VARCHAR(10) NOT NULL,
    Class VARCHAR(20),
    FlightNo VARCHAR(50) NOT NULL,
    FlightDepDate DATE NOT NULL,
    FOREIGN KEY (FlightNo, FlightDepDate) REFERENCES ScheduledFlight(FlightNo, DepDate) on delete cascade
);


------------------------------------------------
--  DDL Statements for table Baggage
------------------------------------------------
CREATE TABLE Baggage (
    TicketNo VARCHAR(50) NOT NULL,
    BagType VARCHAR(50) NOT NULL,
    NumBags INT NOT NULL,
    TotalWeight DECIMAL(5,2),
    Fragile BOOLEAN,
    PRIMARY KEY (TicketNo, BagType),
    FOREIGN KEY (TicketNo) REFERENCES Ticket(TicketNo) on delete cascade
);

------------------------------------------------
--  DDL Statements for table Use
------------------------------------------------
CREATE TABLE Use (
    AirlineAlias VARCHAR(50) NOT NULL,
    RouteID INT NOT NULL,
    PRIMARY KEY (AirlineAlias, RouteID),
    FOREIGN KEY (AirlineAlias) REFERENCES Airline(Alias) on delete cascade,
    FOREIGN KEY (RouteID) REFERENCES Route(RouteID) on delete cascade
);

------------------------------------------------
--  DDL Statements for table Book
------------------------------------------------
CREATE TABLE Book (
    PassengerID INT NOT NULL,
    TicketNo VARCHAR(50) NOT NULL PRIMARY KEY,
    Price DECIMAL(10,2),
    Website VARCHAR(100),
    FOREIGN KEY (PassengerID) REFERENCES Passenger(PersonID) on delete cascade,
    FOREIGN KEY (TicketNo) REFERENCES Ticket(TicketNo) on delete cascade
);

------------------------------------------------
--  DDL Statements for table Flies
------------------------------------------------
CREATE TABLE Flies (
    PilotID INT NOT NULL,
    AirplaneSNo VARCHAR(50) NOT NULL,
    PRIMARY KEY (PilotID, AirplaneSNo),
    FOREIGN KEY (PilotID) REFERENCES Pilot(PersonID),
    FOREIGN KEY (AirplaneSNo) REFERENCES Airplane(SerialNo)
);
