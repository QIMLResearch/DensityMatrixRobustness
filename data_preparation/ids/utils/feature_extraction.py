import subprocess
import os
import platform
import sys
import csv

def get_tshark_path():
    return '/Applications/Wireshark.app/Contents/MacOS/tshark'

def pcap2csv_with_tshark(input_path, output_path):
    tshark_path = get_tshark_path()
    if not tshark_path:
        print("tshark not found.")
        sys.exit(1)

    fields = "-e frame.time_epoch -e frame.len " \
            "-e eth.src -e eth.dst -e eth.type " \
            "-e ip.src -e ip.dst -e ip.proto -e ip.ttl -e ip.version -e ip.flags -e ip.id -e ip.len " \
            "-e ipv6.src -e ipv6.dst " \
            "-e tcp.srcport -e tcp.dstport -e tcp.flags.str -e tcp.seq -e tcp.len -e tcp.window_size " \
            "-e udp.srcport -e udp.dstport -e udp.length " \
            "-e icmp.type -e icmp.code " \
            "-e arp.opcode -e arp.src.hw_mac -e arp.src.proto_ipv4 -e arp.dst.hw_mac -e arp.dst.proto_ipv4 "
              
    cmd = [tshark_path, '-r', input_path, '-T', 'fields', '-E', 'separator=/t', '-E', 'header=y', '-E', 'occurrence=f'] + fields.split()
    
    print(f"Executing command: {' '.join(cmd)}")

    try:
        with open(output_path, 'w') as output_file:
            process = subprocess.Popen(cmd, stdout=output_file, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()
            if stderr:
                print("Error:", stderr)
            elif process.returncode != 0:
                print(f"tshark process returned with error code {process.returncode}.")
            else:
                print("tshark parsing complete. File saved as:", output_path)
    except subprocess.TimeoutExpired:
        print(f"tshark process timed out after 1 hour.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features with tshark")
    parser.add_argument('-i', '--input', required=True, help="Input pcap file path")
    parser.add_argument('-o', '--output', required=True, help="Output CSV file path")
    args = parser.parse_args()

    pcap2csv_with_tshark(args.input, args.output)