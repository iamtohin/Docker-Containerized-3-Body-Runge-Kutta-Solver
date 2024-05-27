import asyncio
import socket
import websockets
import json

async def backend_client():
	HOST = socket.gethostbyname('solver_api')
	PORT = 8001  
	uri = "ws://"+str(HOST)+":"+str(PORT)
	
	async with websockets.connect(uri) as websocket:	

		#Initializing a dummy data structure of spected request

		t1 = 0
		t2 =30
		m1 =5
		m2 =3
		m3 =4
		x1 =1
		x2 =1
		x3 =-2
		y1 =-1
		y2 =3
		y3 =-1
		z1 =0
		z2 =0
		z3 =0
		vx1 =0
		vx2 =0
		vx3 =0
		vy1 =0
		vy2 =0
		vy3 =0
		vz1 =0
		vz2 =0
		vz3 =0
		h =1
		e =1e-9
		s =0.9
		timestamp= 42.42

		data = {
			"Id": timestamp,
			"b1": {
				"position": [x1,y1,z1],
				"velocity": [vx1,vy1,vz1],
				"mass": m1
				},
			"b2": {
				"position": [x2,y2,z2],
				"velocity": [vx2,vy2,vz2],
				"mass": m2
				},
			"b3": {
				"position": [x3,y3,z3],
				"velocity": [vx3,vy3,vz3],
				"mass": m3
				},
			"time": [t1,t2],
			"epsilon":e,
			"h":h,
			"s":s
		}

		
		data=json.dumps(data)

		await websocket.send(data)

		initial = await websocket.recv()
		print(initial)

		print(uri)

		print(json.loads(json.dumps(data)))



asyncio.get_event_loop().run_until_complete(backend_client())




