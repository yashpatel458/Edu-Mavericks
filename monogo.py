from pymongo import MongoClient
import csv

# Connect to MongoDB
client = MongoClient('mongodb://127.0.0.1:27017/')
# Create the database
db = client['AdtProject']
# Create the education collection
education_collection = db['education']

# Create the employmentForecast collection
employment_forecast_collection = db['employmentForecast']

# Create the wagesByEducationalLevel collection
wages_by_educational_level_collection = db['wagesByEducationalLevel']

# Open the CSV file
with open('education_level_region.csv', 'r') as file:
    # Read the CSV data
    csv_data = csv.reader(file)
    
    # Skip the header row
    next(csv_data)
    
    # Iterate over each row in the CSV data
    for row in csv_data:
        # Create a dictionary to store the data
        data = {
            'region': row[0],
            'education_level': row[1],
            'enrollment': row[2],
            'graduation': row[3]
        }
        
        # Insert the data into the education collection
        result = education_collection.insert_one(data)

# Open the CSV file for wages by educational level
with open('wage_rates_by_education.csv', 'r') as file:
    # Read the CSV data
    csv_data = csv.reader(file)

    # Skip the header row
    next(csv_data)

    # Iterate over each row in the CSV data
    for row in csv_data:
        # Create a dictionary to store the data
        data = {
            'education_level': row[0],
            'wage_rate': row[1]
        }
        
        # Insert the data into the wagesByEducationalLevel collection
        result = wages_by_educational_level_collection.insert_one(data)

        # Break the loop after inserting 100 lines of data
        if result.inserted_id == 100:
            break

# Get all data from the education collection
all_data = education_collection.find()

# Iterate over each document in the collection
for data in all_data:
    # Print the data
    print(data)

# Get all data from the education collection
all_data = wages_by_educational_level_collection.find()

# Iterate over each document in the collection
for data in all_data:
    # Print the data
    print(data)
# Close the MongoDB connection
client.close()