---
id: "tiexin-guo-project-dive"
tags: []
---

Full stack -> DevOps -> Join startup

I was nervous

Coding

- Resources metrics
  - GPUs and Network IO

- Struggling with metrics for data structure? data structure?

- cron job <- number of requests in progress

- need to check the requests?
- store request id

q2: distribution

list to store all process time

fixed length list

Implement a gauge for measuring requests in progress?

Class Middleware:

App: asgi_app Def **init**(self): Requests_in_process = set()

def **call**(self, req, receive, send): receive() app.do_something(req) send()
Requests_in_process.add(req.id) self.app(receive, send)
Requests_in_process.delete(req.id)

Def requests_in_process(): Return len(self.xxx)

latency metrics: p50 , p95, p99 over time

Def **call**: Start = time() … app code … End = time() Q = deque() list.sort()
list.append(end-start) P50 = list[len(list)/2]

Distribution (over time?) ?

0-100ms 100-1000ms

Boundaries = [0, 100, 1000, …]

def **call**(): Start = time() … app code … End = time() Time_used = end-start()
Counter = get_list_in_redis_with_timeout() # initialized to all zeroes

For i in range(len(boundaries)): If time_used < boundary[i]: counter[i]+=1

Return [ counter[i]/num_of_requests for i in range(len(counter)) ]
