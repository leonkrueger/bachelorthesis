# Set container name variables
image_name="bachelorthesis_leon_krueger_pyton"
network_name="bachelorthesis_leon_krueger_net"
mysql_container_name="bachelorthesis_leon_krueger_mysql"

# Build and run python container
docker build -t $image_name .
docker run -it --rm --net $network_name -v $(pwd)/evaluation:/mounted_evaluation $image_name