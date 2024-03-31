# Use the official MongoDB image from Docker Hub
FROM mongo:latest

# Expose the default MongoDB port
EXPOSE 27017

# Set the default command for the container
CMD ["mongod"]