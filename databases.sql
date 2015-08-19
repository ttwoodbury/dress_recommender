CREATE TABLE clothes (
	id serial PRIMARY KEY,
	title varchar, 
	color varchar, 
	price real,
	source varchar, 
	url varchar );

CREATE TABLE images (
	id serial PRIMARY KEY, 
	dress_id int references clothes(id), 
	file_name varchar, 
	url varchar);

CREATE TABLE image_features (
	img_id int references images(id),
	gl_features real[]);

CREATE TABLE image_features2 (
	img_id int,
	gl_features real[],
	original_gl real[],
	color_features real[],
	texture_features real[],
	max_color int[]);

ALTER TABLE images 
	ADD CONSTRAINT dress_id FOREIGN KEY clothes(id);