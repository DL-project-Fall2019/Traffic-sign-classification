#!/bin/bash
# Created by Yufeng Yang 11/24/2019
# Bash Script for filtering the classes that contains pictures less than fivei and merge same class together
# Run this script at data/../
cd data/
rm -rf tt100
mkdir tt100

mv tt100k_sign_only.7z tt100
7za x belgium_ts.7z
rm -f belgium_ts.7z
mv tt100k_style belgium
cd tt100
7za x tt100k_sign_only.7z
rm -f tt100k_sign_only.7z
cd ..
# now at data/
rm -f belgium_name.txt
rm -f tt100_name.txt
cd belgium
ls > ../belgium_name.txt
cd ../tt100
ls > ../tt100_name.txt
cd ..
# now at data/
rm -f filter_belgium.txt
rm -f filter_tt100.txt
for class in $(cat belgium_name.txt); do
	num=$(ls belgium/${class}/ | wc -l)
	if [ $num -ge 5 ]; then
		echo $class >> filter_belgium.txt
	fi
done

for class in $(cat tt100_name.txt); do
	num=$(ls tt100/${class}/ | wc -l)
	if [ $num -ge 5 ]; then
                echo $class >> filter_tt100.txt
	fi
done

cat filter_belgium.txt filter_tt100.txt > filter_name.txt
sort filter_name.txt | uniq > final_class.txt
rm -f filter_*
rm -f *_name.txt

for class in $(cat final_class.txt); do
	rm -rf $class
	mkdir $class
	cp belgium/${class}/* ${class}/ 2>/dev/null
	cp tt100/${class}/* ${class}/ 2>/dev/null
done
echo =======Data Preparation Done=======
rm -f final_class.txt
rm -rf belgium
rm -rf tt100
