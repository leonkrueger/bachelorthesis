# Set container name variables
image_name="bachelorthesis_leon_krueger_pyton"
network_name="bachelorthesis_leon_krueger_net"
mysql_container_name="bachelorthesis_leon_krueger_mysql"

# Remove existing SQL containers and images
docker container stop $mysql_container_name
docker network rm $network_name
docker rm $mysql_container_name

# Create network and run SQL container
docker network create $network_name
docker run -d --name $mysql_container_name --net $network_name -p 3306:3306 -e "MYSQL_ROOT_PASSWORD=Baum_4392" -e "MYSQL_DATABASE=db" -e "MYSQL_USER=user" -e "MYSQL_PASSWORD=Start123" -d mysql

sleep 120

# Build and run python container
cd ..
docker build -t $image_name -f evaluation/bachelorthesis/Dockerfile .
docker run -it --rm --net $network_name -v $(pwd)/evaluation/bachelorthesis:/app/mounted_evaluation $image_name

# Remove existing SQL containers and images
docker container stop $mysql_container_name
docker network rm $network_name
docker rm $mysql_container_name