#this program helps user to attach any ec2 classic instances to vpc using classic link features
#you should know what is your vpc-id and vpc group security

import boto
from boto import vpc
from boto import ec2
import itertools
from optparse import OptionParser 
grp_name = "<security-grp-name>"
vpc_id = "<vpc-id>"
region="<region>"
conn_vpc = vpc.VPCConnection("<secrete_key>","<access_key>")

#find out the all the instances from a given region and name
def getInstances(groupnames):
	conn_ec2 = ec2.connect_to_region(region)
	print conn_ec2
	reservations = conn_ec2.get_all_reservations(filters={"instance.group-name": groupnames})
	instances = itertools.chain.from_iterable(r.instances for r in reservations)
	return [i.id for i in instances if i.state not in ["shutting-down", "terminated"]]


def attach_link(groupname):
	instances = getInstances(groupname)
	for instance in instances:
		conn_vpc.attach_classic_link_vpc(vpc_id,instance, ["<vpc-security-grp>"], dry_run=False)
	for instance in master:
		conn_vpc.attach_classic_link_vpc(vpc_id,instance, ["<vpc-security-grp>"], dry_run=False)

def detach_link(groupname):
	instances = getInstances(groupname)
	for instance in instances:
		conn_vpc.detach_classic_link_vpc(vpc_id,instance, dry_run=False)
	for instance in master:
		conn_vpc.detach_classic_link_vpc(vpc_id,instance, dry_run=False)

def main():
	(opts, args) = parse_arg()
	if (opts.attach):
		attach_link(args[0])
	elif (opts.detach):
		detach_link(args[0])

def parse_arg():
	parser = OptionParser(prog="link-ec2")
	parser.add_option(
        "-a", "--attach", action="store_true", default=False,
        help="attach an EC2 instance to vpc through classic link")
	parser.add_option(
        "-d", "--detach", action="store_true", default=False,
        help="detach an EC2 instance from vpc")
	(opts, args) = parser.parse_args()
	return (opts, args)

if __name__ =="__main__":
	main()