# Set container name variables
image_name="bachelorthesis_leon_krueger_pyton"
network_name="bachelorthesis_leon_krueger_net"
mysql_container_name="bachelorthesis_leon_krueger_mysql"

# Remove existing SQL containers and images
docker container stop $mysql_container_name
docker network rm $network_name
docker rm $mysql_container_name
