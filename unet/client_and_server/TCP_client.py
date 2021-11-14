import socket, time

dest_ip = ('192.168.31.56', 13579)
UDP_server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
UDP_server.connect(dest_ip)
example_image_path = './UDP_Files/Recv/test.jpg'
example_save_path = './UDP_Files/Send/test_result.jpg'


def send_image(image_path):
    with open(image_path, 'rb') as f:
        data = f.read()
    len_data = len(data)
    space = len_data
    UDP_server.sendall(space.to_bytes(16, 'big'))
    UDP_server.sendall(data)
    # for i in range(0, space):
    #     start_idx = i * 10000
    #     end_idx = min(len_data, (i+1) * 10000)
    #     UDP_server.sendall(data[start_idx: end_idx])
    #     print('Send {}'.format(end_idx - start_idx))
    return 'Send Image {} OK'.format(image_path)


def recv_image(image_path):
    space = UDP_server.recv(16)
    space = int.from_bytes(space, 'big')
    print('space {}'.format(space))
    with open(image_path, 'wb') as f:
        while space > 0:
            data = UDP_server.recv(1024)
            print('data len {}'.format(len(data)))
            space -= len(data)
            f.write(data)
    return 'Recv Image {} OK'.format(image_path)


while True:
    print(send_image(example_image_path))
    print(recv_image(example_save_path))
    print('Send OK!')
    time.sleep(2)