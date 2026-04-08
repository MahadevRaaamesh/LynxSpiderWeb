# Create a 2x2 grid network
netgenerate --grid --grid.number 2 --grid.length 200 --output-file map.net.xml

# Generate random trips for the network
python "%SUMO_HOME%\tools\randomTrips.py" -n map.net.xml -e 1000 -p 1 --route-file map.rou.xml
