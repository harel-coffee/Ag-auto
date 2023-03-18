
cat /data/kiran/Data/VIC/params/metdatalistmultipleset/list_set1 | while read LINE ; do
echo "copying $LINE..."
cp /data/kirti/vic_inputdata0625_pnw_combined_05142008/$LINE ./metdata/
done

#####################

dir_list=()
while IFS= read -d $'\0' -r file ; do
dir_list=("${dir_list[@]}" "$file")
done < <(find /data/hydro/jennylabcommon2/metdata/maca_v2_vic_binary/ -mindepth 2 -maxdepth 2 -type d -print0)


dir_list=()
cat /Users/hn/Documents/GitHub/Kirti/analogy/parameters/missing_locations_short | while read LINE ; do
dir_list=("${dir_list[@]}" "$LINE")
printf '%s\n' "${dir_list[@]}"
done


dir_list=()
cat /home/hnoorazar/analog_codes/parameters/file_names | while read LINE ; do
dir_list=("${dir_list[@]}" "$LINE")
printf '%s\n' "${dir_list[@]}"
done


