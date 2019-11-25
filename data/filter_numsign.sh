#~/bin/bash!
# Created by Yufeng Yang 11/24/2019
# Bash Script for filtering the classes that contains pictures less than five
# Need: A txt file that contains the name of each class.
rm -f filtered_belgium_list.txt
# directory=tt100k
directory=belgium/tt100k_style
for class in $(cat belgium.txt); do
	num=$(ls ${directory}/${class}/ | wc -l)
	if [ $num -ge 5 ]; then
		echo $class >> filtered_belgium_list.txt
	fi
done
