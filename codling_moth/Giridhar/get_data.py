import os

#histData = '/data/hydro/jennylabcommon2/metdata/historical/UI_historical/VIC_Binary_CONUS'

def getLocations(loc_path, loc_file):
	locations = []
	with open(loc_path + "/" + loc_file, 'r') as file:
		lines = file.readlines()
		for line in lines:
			locations.append(line.strip())
	return locations

def getFuture(locations):
	#futureData = '/data/hydro/jennylabcommon2/metdata/VIC_ext_maca_v2_binary_westUSA'
	futureData = '/data/hydro/jennylabcommon2/metdata/maca_v2_vic_binary'
	futures = ["BNU-ESM", "CanESM2", "GFDL-ESM2G", "bcc-csm1-1-m", "CNRM-CM5", "GFDL-ESM2M"]
	#futures = ["CanESM2", "GFDL-ESM2G", "bcc-csm1-1-m", "CNRM-CM5", "GFDL-ESM2M"]
	for future in futures:
		for version in ['rcp45', 'rcp85']:
			if not os.path.exists(os.getcwd() + "/data/" + future + "/" + version):
				os.makedirs(os.getcwd() + "/data/" + future + "/" + version)
			new_directory = os.getcwd() + "/data/" + future + "/" + version
			#i = 1
			for location in locations:
				location = location[:-2]
				location += "*"
				file_name = futureData + "/" + future + "/" + version + "/" + "data_" + location
				command = "cp " + file_name + " " + new_directory
				print command
				#i += 1
				#print i
				os.system(command)

def getHistorical(locations):
	histData = '/data/hydro/jennylabcommon2/metdata/historical/UI_historical/VIC_Binary_CONUS/'
	new_directory = os.getcwd() + "/data/historical/"
	if not os.path.exists(new_directory):
		os.makedirs(new_directory)
	for location in locations:
		file_name = histData + "data_" + location
		command = "cp " + file_name + " " + new_directory
		print command
		os.system(command)

locations = getLocations(os.getcwd(), "all_us_locations_list")
#getFuture(locations)
getHistorical(locations)
