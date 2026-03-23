FILE=./multi_core_imbalanced/stats.txt

echo $FILE

metrics_l1=("M.Load::total"  "E.Load::total"  "S.Load::total" "I.Load::total")
metrics_l2=("L2Cache_Controller.L1_GETS" "L2Cache_Controller.L1_GETX" )
metrics_network=( "network.msg_count.Request_Control" "network.msg_count.Response_Data" "network.msg_count.Writeback_Data" )


#metrics=("readBW" "writeBW" "ReadReq.misses" "avgMissLatency")
echo " "
echo "L1 Cache controller stats"
for metric in "${metrics_l1[@]}"; do
    echo "-------------------------------"
    grep -ri "$metric" $FILE
done
echo " "

echo "L2 Cache controller stats"
for metric in "${metrics_l2[@]}"; do
    echo "-------------------------------"
    grep -ri "$metric" $FILE
done
echo " "
echo "Network stats"
for metric in "${metrics_network[@]}"; do
    echo "-------------------------------"
    grep -ri "$metric" $FILE
done


