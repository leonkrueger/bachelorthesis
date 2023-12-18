# Set container name variables
image_name="bachelorthesis_leon_krueger_pyton"
network_name="bachelorthesis_leon_krueger_net"
mysql_container_name="bachelorthesis_leon_krueger_mysql"

# Run python container
docker run -it --rm --net $network_name $image_name