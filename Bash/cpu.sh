#check linux info

#查看物理 cpu 数：
cat /proc/cpuinfo| grep "physical id"| sort| uniq| wc -l

#查看每个物理 cpu 中 核心数(core 数)：
cat /proc/cpuinfo | grep "cpu cores" | uniq

#查看总的逻辑 cpu 数（processor 数）：
cat /proc/cpuinfo| grep "processor"| wc -l

#查看 cpu 型号：
cat /proc/cpuinfo | grep name | cut -f2 -d: | uniq -c

#判断 cpu 是否 64 位：
#check cpuinfo 中的 flags 区段，看是否有 lm （long mode） 标识

# Check cpu information
lscpu
#...
#CPU(s):                24
#On-line CPU(s) list:   0-23
#Thread(s) per core:    2
#Core(s) per socket:    6
#Socket(s):             2
#...

#一台完整的计算机可能包含一到多个物理 cpu
#从单个物理 cpu （physical cpu）的角度看，其可能是单核心、双核心甚至多核心的
#从单个核心（core）的角度看，还有 SMT / HT 等技术让每个 core 对计算机操作系统而言用起来像多个物理 core 差不多


#总的逻辑 cpu 数 = 物理 cpu 数 * 每颗物理 cpu 的核心数 * 每个核心的超线程数

