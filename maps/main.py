import xml.etree.ElementTree as ET

INPUT_FILE = "Dynamic-Traffic-light-management-system-main/maps/city1.rou.xml"
OUTPUT_FILE = "Dynamic-Traffic-light-management-system-main/maps/city1_updated.rou.xml"

tree = ET.parse(INPUT_FILE)
root = tree.getroot()

for vehicle in root.findall("vehicle"):
    vid = vehicle.get("id")

    # Skip if type already exists
    if vehicle.get("type") is not None:
        continue

    try:
        vid_int = int(vid)
        if vid_int % 7 == 0:
            vehicle.set("type", "ambulance")
        else:
            vehicle.set("type", "car")
    except:
        vehicle.set("type", "car")

tree.write(OUTPUT_FILE, encoding="utf-8", xml_declaration=True)
print("✅ Updated file written as:", OUTPUT_FILE)