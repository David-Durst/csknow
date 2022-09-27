#include <vector>

static inline __attribute__((always_inline))
int64_t positiveModulo(int64_t x, int64_t y) {
    return ((x % y) + y) % y;
}

// based on https://gist.github.com/edwintcloud/d547a4f9ccaf7245b06f0e8782acefaa
template <class T>
class CircularBuffer {
    std::vector<T> buffer;
    int64_t head = 0, tail = 0, curSize = 0;

public:
    CircularBuffer<T>(int64_t max_size) : buffer(max_size) { }

    inline int64_t maxSize() const { return buffer.size(); }
    inline int64_t getCurSize() const { return curSize; }

    // https://embeddedartistry.com/blog/2017/05/17/creating-a-circular-buffer-in-c-and-c/
    // using approach where tracking full size
    inline bool isFull() const { return tail == head && curSize > 0; }
    inline bool isEmpty() const { return tail == head && curSize == 0; }
    void fill(T element) {
        while (!isFull()) {
            enqueue(element, true);
        }
    }

    T & fromOldest(int64_t i = 0) { return buffer[(head + i) % maxSize()]; }
    const T & fromOldest(int64_t i = 0) const { return buffer[(head + i) % maxSize()]; }
    T & fromNewest(int64_t i = 0) { return buffer[positiveModulo(tail - i - 1, maxSize())]; }
    const T & fromNewest(int64_t i = 0) const { return buffer[positiveModulo(tail - i - 1, maxSize())]; }
    const std::vector<T> & getVector() { return buffer; }

    void clear() { 
        head = tail;
        curSize = 0;
    }

    void enqueue (T item, bool overwrite = true) {
        if (isFull()) {
            if (overwrite) {
                dequeue();
            }
            else {
                throw std::runtime_error("buffer is full, not using overwrite");
            }
        }

        curSize++;
        buffer[tail] = item;
        tail = (tail + 1) % maxSize();
    }

    T dequeue() {
        if (isEmpty()) {
            throw std::runtime_error("buffer is empty");
        }

        curSize--;
        T result = buffer[head];
        head = (head + 1) % maxSize();
        return result;
    }
};
