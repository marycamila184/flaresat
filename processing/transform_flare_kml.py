import xml.etree.ElementTree as ET
import csv

tree = ET.parse('/home/marycamila/flaresat/source/flare_points/energies-09-00014-s001.kml')
root = tree.getroot()

# Define namespaces
namespaces = {'kml': 'http://www.opengis.net/kml/2.2'}

# Open the CSV file for writing
with open('/home/marycamila/flaresat/source/csv_points/gas_flaring_points.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(['Rank', 'Country', 'Flare ID', 'Latitude', 'Longitude', 'Temperature', 'RadiativeHeat', 'Frequency', 'Area', 'BCM', 'Type', 'queue'])

    # Iterate over each Placemark element in the KML file
    for placemark in root.findall('.//kml:Placemark', namespaces):
        rank = placemark.find('kml:name', namespaces).text
        description = placemark.find('kml:description', namespaces).text
        
        # Extract the values from the description table
        country = description.split('Country: <b>')[1].split('</b>')[0]
        flare_id = description.split('Flare ID: <b>')[1].split('</b>')[0]
        lat = description.split('Lat=')[1].split(', ')[0]
        lon = description.split('Lon=')[1].split(' deg.')[0]
        temp = description.split('Tavg=')[1].split(' K')[0]
        rh = description.split('RHsum=')[1].split(' MW')[0]
        freq = description.split('Freq. detect.=')[1].split(' %')[0]
        area = description.split('Area=')[1].split(' m2')[0]
        bcm = description.split('BCM=')[1].split('</td>')[0]
        flare_type = description.split('Type: ')[1].split('</td>')[0]

        # Write the row to the CSV file
        writer.writerow([rank, country, flare_id, lat, lon, temp, rh, freq, area, bcm, flare_type, True])
