connect to se3db3;

create table cellphone(
  maker char(20),
  model char(20),
  version int,
  price decimal(10,2)
);

insert into cellphone (maker, model, version, price) values
('Apple', 'iPhone 13', 1, 999.99),
('Samsung', 'Galaxy S21', 2, 799.99),
('Google', 'Pixel 6', 1, 699.99);

create table playlist(
  artist char(20),
  album char(20),
  song char(20),
  released int
);

insert into playlist (artist, album, song, released) values
('Ed Sheeran', '÷ (Divide)', 'Shape of You', 2017),
('Adele', '25', 'Hello', 2015),
('Taylor Swift', '1989', 'Shake It Off', 2014);
