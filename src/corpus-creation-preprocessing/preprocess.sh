#!/usr/bin/env bash

corpus=$1 # The text corpus to process
folder=$2 # Folder which contains all files created during script (will be auto-deleted)
prefix=$3 # DB File prefix

echo "Corpus: "$corpus", Folder: "$folder", DB File prefix: "$prefix

n=`grep "" -c $corpus | awk '{ print $1 }'`

m=15
n=$((($n + 1)/$m))
echo "Splitting corpus into "$m" parts" 

min_frequency=7 # Threshold for considering most frequent paths. Chosen after optimization.
max_path_length=10 # Maximum path length considered while extracting paths. Chosen after optimization.

parts=( $(seq 1 $m ) )

echo -e "\n\nTunable Parameters:\n\nPath Frequency: "$min_frequency"\nMax Length of paths: " $max_path_length"\n\n"

echo "Stage 1/3 : Splitting corpus..."

split -l $n $corpus $corpus"_split_" --numeric-suffixes=1;

echo "Stage 2/3 : Parsing corpus..."

for x in "${parts[@]}"
do
	corpus_part=$corpus"_split_"$x
	( python3 corpus_parser.py $corpus_part ) &
done
wait

echo "Stage 3/3: The main stage: tuning of parameters..."

parsed_final=$corpus"_"$max_path_length"_parsed"
cat $corpus"_split_"*"_"$max_path_length"_parsed" > $parsed_final

echo -e "\tStep: Counting relations..."
for x in "${parts[@]}"
do
	parsed_final_part=$corpus"_split_"$x"_"$max_path_length"_parsed"
	( awk -F "\t" '{relations[$3]++} END{for(relation in relations){print relation"\t"relations[relation]}}' $parsed_final_part > $corpus"_paths_"$x"_"$max_path_length ) &
done
wait

paths=$folder"all_paths_"$max_path_length
cat $corpus"_paths_"*"_"$max_path_length > $paths
rm $corpus"_paths_"*"_"$max_path_length

echo -e "\tStep: Filtering common paths..."

for n in "${min_frequencys[@]}"
do
	echo -e "\t\tPath threshold: "$n
	awk -F "\t" '{i[$1]+=$2} END{for(x in i){ if (i[x] >= '$n') print x } }' $paths > $folder'filtered_paths' 
done
rm $paths

echo -e "\tStep: Creating word files..."
awk -F$'\t' '{if (a[$1] == 0) {a[$1] = -1; print $1}}' $parsed_final > $folder"xterms_"$max_path_length & PIDLEFT=$!
awk -F$'\t' '{if (a[$2] == 0) {a[$2] = -1; print $2}}' $parsed_final > $folder"yterms_"$max_path_length & PIDRIGHT=$!

wait $PIDLEFT
wait $PIDRIGHT
cat $folder"xterms_"$max_path_length $folder"yterms_"$max_path_length | sort -u --parallel=128 > $folder"terms_"$max_path_length;
rm $folder"xterms_"$max_path_length $folder"yterms_"$max_path_length $parsed_final

echo -e "\tStep: Creating term and path db files..."
for n in "${min_frequencys[@]}"
do
	( python3 create_db_files.py $folder $folder"terms_"$max_path_length $prefix 1; ) &
done
wait

rm $folder"terms_"$max_path_length $folder"filtered_paths"

echo -e "\tStep: Processing triplets..."

echo -e "\t\tPath threshold: "$n

# Creating an ID file for the parsed triplets
for x in "${parts[@]}"
do
	parsed_final_part=$corpus"_split_"$x"_"$max_path_length"_parsed"
	( python3 create_db_files.py $folder $parsed_final_part $prefix 2; ) &
done
wait

# Counting triplet IDs to calculate number of occurences
for x in "${parts[@]}"
do
	triplet_part_file=$folder"triplet_id_"$x
	triplet_count_file=$folder"triplet_count_"$x
	( awk -F "\t" '{relations[$0]++} END{for(relation in relations){print relation"\t"relations[relation]}}' $triplet_part_file > $triplet_count_file ) &
done
wait

rm $folder"triplet_id_"*

cat $folder"triplet_count_"* > $folder"triplet_count";

rm $folder"triplet_count_"*
# Creating a triplet occurence matrix
sort -t$'\t' -k1 -n --parallel=128 $folder"triplet_count" > $folder"triplet_sorted"
rm $folder"triplet_count"

python3 create_db_files.py $folder $folder"triplet_sorted" $prefix 3;

rm $folder"triplet_sorted"


rm $corpus"_split_"*