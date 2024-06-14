import { h, r as registerInstance, c as createEvent, g as getElement, H as Host } from './index-b27121e8.js';

const global$1 = (typeof global !== "undefined" ? global :
  typeof self !== "undefined" ? self :
  typeof window !== "undefined" ? window : {});

var lookup = [];
var revLookup = [];
var Arr = typeof Uint8Array !== 'undefined' ? Uint8Array : Array;
var inited = false;
function init () {
  inited = true;
  var code = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
  for (var i = 0, len = code.length; i < len; ++i) {
    lookup[i] = code[i];
    revLookup[code.charCodeAt(i)] = i;
  }

  revLookup['-'.charCodeAt(0)] = 62;
  revLookup['_'.charCodeAt(0)] = 63;
}

function toByteArray (b64) {
  if (!inited) {
    init();
  }
  var i, j, l, tmp, placeHolders, arr;
  var len = b64.length;

  if (len % 4 > 0) {
    throw new Error('Invalid string. Length must be a multiple of 4')
  }

  // the number of equal signs (place holders)
  // if there are two placeholders, than the two characters before it
  // represent one byte
  // if there is only one, then the three characters before it represent 2 bytes
  // this is just a cheap hack to not do indexOf twice
  placeHolders = b64[len - 2] === '=' ? 2 : b64[len - 1] === '=' ? 1 : 0;

  // base64 is 4/3 + up to two characters of the original data
  arr = new Arr(len * 3 / 4 - placeHolders);

  // if there are placeholders, only get up to the last complete 4 chars
  l = placeHolders > 0 ? len - 4 : len;

  var L = 0;

  for (i = 0, j = 0; i < l; i += 4, j += 3) {
    tmp = (revLookup[b64.charCodeAt(i)] << 18) | (revLookup[b64.charCodeAt(i + 1)] << 12) | (revLookup[b64.charCodeAt(i + 2)] << 6) | revLookup[b64.charCodeAt(i + 3)];
    arr[L++] = (tmp >> 16) & 0xFF;
    arr[L++] = (tmp >> 8) & 0xFF;
    arr[L++] = tmp & 0xFF;
  }

  if (placeHolders === 2) {
    tmp = (revLookup[b64.charCodeAt(i)] << 2) | (revLookup[b64.charCodeAt(i + 1)] >> 4);
    arr[L++] = tmp & 0xFF;
  } else if (placeHolders === 1) {
    tmp = (revLookup[b64.charCodeAt(i)] << 10) | (revLookup[b64.charCodeAt(i + 1)] << 4) | (revLookup[b64.charCodeAt(i + 2)] >> 2);
    arr[L++] = (tmp >> 8) & 0xFF;
    arr[L++] = tmp & 0xFF;
  }

  return arr
}

function tripletToBase64 (num) {
  return lookup[num >> 18 & 0x3F] + lookup[num >> 12 & 0x3F] + lookup[num >> 6 & 0x3F] + lookup[num & 0x3F]
}

function encodeChunk (uint8, start, end) {
  var tmp;
  var output = [];
  for (var i = start; i < end; i += 3) {
    tmp = (uint8[i] << 16) + (uint8[i + 1] << 8) + (uint8[i + 2]);
    output.push(tripletToBase64(tmp));
  }
  return output.join('')
}

function fromByteArray (uint8) {
  if (!inited) {
    init();
  }
  var tmp;
  var len = uint8.length;
  var extraBytes = len % 3; // if we have 1 byte left, pad 2 bytes
  var output = '';
  var parts = [];
  var maxChunkLength = 16383; // must be multiple of 3

  // go through the array every three bytes, we'll deal with trailing stuff later
  for (var i = 0, len2 = len - extraBytes; i < len2; i += maxChunkLength) {
    parts.push(encodeChunk(uint8, i, (i + maxChunkLength) > len2 ? len2 : (i + maxChunkLength)));
  }

  // pad the end with zeros, but make sure to not forget the extra bytes
  if (extraBytes === 1) {
    tmp = uint8[len - 1];
    output += lookup[tmp >> 2];
    output += lookup[(tmp << 4) & 0x3F];
    output += '==';
  } else if (extraBytes === 2) {
    tmp = (uint8[len - 2] << 8) + (uint8[len - 1]);
    output += lookup[tmp >> 10];
    output += lookup[(tmp >> 4) & 0x3F];
    output += lookup[(tmp << 2) & 0x3F];
    output += '=';
  }

  parts.push(output);

  return parts.join('')
}

function read (buffer, offset, isLE, mLen, nBytes) {
  var e, m;
  var eLen = nBytes * 8 - mLen - 1;
  var eMax = (1 << eLen) - 1;
  var eBias = eMax >> 1;
  var nBits = -7;
  var i = isLE ? (nBytes - 1) : 0;
  var d = isLE ? -1 : 1;
  var s = buffer[offset + i];

  i += d;

  e = s & ((1 << (-nBits)) - 1);
  s >>= (-nBits);
  nBits += eLen;
  for (; nBits > 0; e = e * 256 + buffer[offset + i], i += d, nBits -= 8) {}

  m = e & ((1 << (-nBits)) - 1);
  e >>= (-nBits);
  nBits += mLen;
  for (; nBits > 0; m = m * 256 + buffer[offset + i], i += d, nBits -= 8) {}

  if (e === 0) {
    e = 1 - eBias;
  } else if (e === eMax) {
    return m ? NaN : ((s ? -1 : 1) * Infinity)
  } else {
    m = m + Math.pow(2, mLen);
    e = e - eBias;
  }
  return (s ? -1 : 1) * m * Math.pow(2, e - mLen)
}

function write (buffer, value, offset, isLE, mLen, nBytes) {
  var e, m, c;
  var eLen = nBytes * 8 - mLen - 1;
  var eMax = (1 << eLen) - 1;
  var eBias = eMax >> 1;
  var rt = (mLen === 23 ? Math.pow(2, -24) - Math.pow(2, -77) : 0);
  var i = isLE ? 0 : (nBytes - 1);
  var d = isLE ? 1 : -1;
  var s = value < 0 || (value === 0 && 1 / value < 0) ? 1 : 0;

  value = Math.abs(value);

  if (isNaN(value) || value === Infinity) {
    m = isNaN(value) ? 1 : 0;
    e = eMax;
  } else {
    e = Math.floor(Math.log(value) / Math.LN2);
    if (value * (c = Math.pow(2, -e)) < 1) {
      e--;
      c *= 2;
    }
    if (e + eBias >= 1) {
      value += rt / c;
    } else {
      value += rt * Math.pow(2, 1 - eBias);
    }
    if (value * c >= 2) {
      e++;
      c /= 2;
    }

    if (e + eBias >= eMax) {
      m = 0;
      e = eMax;
    } else if (e + eBias >= 1) {
      m = (value * c - 1) * Math.pow(2, mLen);
      e = e + eBias;
    } else {
      m = value * Math.pow(2, eBias - 1) * Math.pow(2, mLen);
      e = 0;
    }
  }

  for (; mLen >= 8; buffer[offset + i] = m & 0xff, i += d, m /= 256, mLen -= 8) {}

  e = (e << mLen) | m;
  eLen += mLen;
  for (; eLen > 0; buffer[offset + i] = e & 0xff, i += d, e /= 256, eLen -= 8) {}

  buffer[offset + i - d] |= s * 128;
}

var toString = {}.toString;

var isArray = Array.isArray || function (arr) {
  return toString.call(arr) == '[object Array]';
};

/*!
 * The buffer module from node.js, for the browser.
 *
 * @author   Feross Aboukhadijeh <feross@feross.org> <http://feross.org>
 * @license  MIT
 */

var INSPECT_MAX_BYTES = 50;

/**
 * If `Buffer.TYPED_ARRAY_SUPPORT`:
 *   === true    Use Uint8Array implementation (fastest)
 *   === false   Use Object implementation (most compatible, even IE6)
 *
 * Browsers that support typed arrays are IE 10+, Firefox 4+, Chrome 7+, Safari 5.1+,
 * Opera 11.6+, iOS 4.2+.
 *
 * Due to various browser bugs, sometimes the Object implementation will be used even
 * when the browser supports typed arrays.
 *
 * Note:
 *
 *   - Firefox 4-29 lacks support for adding new properties to `Uint8Array` instances,
 *     See: https://bugzilla.mozilla.org/show_bug.cgi?id=695438.
 *
 *   - Chrome 9-10 is missing the `TypedArray.prototype.subarray` function.
 *
 *   - IE10 has a broken `TypedArray.prototype.subarray` function which returns arrays of
 *     incorrect length in some situations.

 * We detect these buggy browsers and set `Buffer.TYPED_ARRAY_SUPPORT` to `false` so they
 * get the Object implementation, which is slower but behaves correctly.
 */
Buffer.TYPED_ARRAY_SUPPORT = global$1.TYPED_ARRAY_SUPPORT !== undefined
  ? global$1.TYPED_ARRAY_SUPPORT
  : true;

function kMaxLength () {
  return Buffer.TYPED_ARRAY_SUPPORT
    ? 0x7fffffff
    : 0x3fffffff
}

function createBuffer (that, length) {
  if (kMaxLength() < length) {
    throw new RangeError('Invalid typed array length')
  }
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    // Return an augmented `Uint8Array` instance, for best performance
    that = new Uint8Array(length);
    that.__proto__ = Buffer.prototype;
  } else {
    // Fallback: Return an object instance of the Buffer class
    if (that === null) {
      that = new Buffer(length);
    }
    that.length = length;
  }

  return that
}

/**
 * The Buffer constructor returns instances of `Uint8Array` that have their
 * prototype changed to `Buffer.prototype`. Furthermore, `Buffer` is a subclass of
 * `Uint8Array`, so the returned instances will have all the node `Buffer` methods
 * and the `Uint8Array` methods. Square bracket notation works as expected -- it
 * returns a single octet.
 *
 * The `Uint8Array` prototype remains unmodified.
 */

function Buffer (arg, encodingOrOffset, length) {
  if (!Buffer.TYPED_ARRAY_SUPPORT && !(this instanceof Buffer)) {
    return new Buffer(arg, encodingOrOffset, length)
  }

  // Common case.
  if (typeof arg === 'number') {
    if (typeof encodingOrOffset === 'string') {
      throw new Error(
        'If encoding is specified then the first argument must be a string'
      )
    }
    return allocUnsafe(this, arg)
  }
  return from(this, arg, encodingOrOffset, length)
}

Buffer.poolSize = 8192; // not used by this implementation

// TODO: Legacy, not needed anymore. Remove in next major version.
Buffer._augment = function (arr) {
  arr.__proto__ = Buffer.prototype;
  return arr
};

function from (that, value, encodingOrOffset, length) {
  if (typeof value === 'number') {
    throw new TypeError('"value" argument must not be a number')
  }

  if (typeof ArrayBuffer !== 'undefined' && value instanceof ArrayBuffer) {
    return fromArrayBuffer(that, value, encodingOrOffset, length)
  }

  if (typeof value === 'string') {
    return fromString(that, value, encodingOrOffset)
  }

  return fromObject(that, value)
}

/**
 * Functionally equivalent to Buffer(arg, encoding) but throws a TypeError
 * if value is a number.
 * Buffer.from(str[, encoding])
 * Buffer.from(array)
 * Buffer.from(buffer)
 * Buffer.from(arrayBuffer[, byteOffset[, length]])
 **/
Buffer.from = function (value, encodingOrOffset, length) {
  return from(null, value, encodingOrOffset, length)
};

if (Buffer.TYPED_ARRAY_SUPPORT) {
  Buffer.prototype.__proto__ = Uint8Array.prototype;
  Buffer.__proto__ = Uint8Array;
}

function assertSize (size) {
  if (typeof size !== 'number') {
    throw new TypeError('"size" argument must be a number')
  } else if (size < 0) {
    throw new RangeError('"size" argument must not be negative')
  }
}

function alloc (that, size, fill, encoding) {
  assertSize(size);
  if (size <= 0) {
    return createBuffer(that, size)
  }
  if (fill !== undefined) {
    // Only pay attention to encoding if it's a string. This
    // prevents accidentally sending in a number that would
    // be interpretted as a start offset.
    return typeof encoding === 'string'
      ? createBuffer(that, size).fill(fill, encoding)
      : createBuffer(that, size).fill(fill)
  }
  return createBuffer(that, size)
}

/**
 * Creates a new filled Buffer instance.
 * alloc(size[, fill[, encoding]])
 **/
Buffer.alloc = function (size, fill, encoding) {
  return alloc(null, size, fill, encoding)
};

function allocUnsafe (that, size) {
  assertSize(size);
  that = createBuffer(that, size < 0 ? 0 : checked(size) | 0);
  if (!Buffer.TYPED_ARRAY_SUPPORT) {
    for (var i = 0; i < size; ++i) {
      that[i] = 0;
    }
  }
  return that
}

/**
 * Equivalent to Buffer(num), by default creates a non-zero-filled Buffer instance.
 * */
Buffer.allocUnsafe = function (size) {
  return allocUnsafe(null, size)
};
/**
 * Equivalent to SlowBuffer(num), by default creates a non-zero-filled Buffer instance.
 */
Buffer.allocUnsafeSlow = function (size) {
  return allocUnsafe(null, size)
};

function fromString (that, string, encoding) {
  if (typeof encoding !== 'string' || encoding === '') {
    encoding = 'utf8';
  }

  if (!Buffer.isEncoding(encoding)) {
    throw new TypeError('"encoding" must be a valid string encoding')
  }

  var length = byteLength(string, encoding) | 0;
  that = createBuffer(that, length);

  var actual = that.write(string, encoding);

  if (actual !== length) {
    // Writing a hex string, for example, that contains invalid characters will
    // cause everything after the first invalid character to be ignored. (e.g.
    // 'abxxcd' will be treated as 'ab')
    that = that.slice(0, actual);
  }

  return that
}

function fromArrayLike (that, array) {
  var length = array.length < 0 ? 0 : checked(array.length) | 0;
  that = createBuffer(that, length);
  for (var i = 0; i < length; i += 1) {
    that[i] = array[i] & 255;
  }
  return that
}

function fromArrayBuffer (that, array, byteOffset, length) {

  if (byteOffset < 0 || array.byteLength < byteOffset) {
    throw new RangeError('\'offset\' is out of bounds')
  }

  if (array.byteLength < byteOffset + (length || 0)) {
    throw new RangeError('\'length\' is out of bounds')
  }

  if (byteOffset === undefined && length === undefined) {
    array = new Uint8Array(array);
  } else if (length === undefined) {
    array = new Uint8Array(array, byteOffset);
  } else {
    array = new Uint8Array(array, byteOffset, length);
  }

  if (Buffer.TYPED_ARRAY_SUPPORT) {
    // Return an augmented `Uint8Array` instance, for best performance
    that = array;
    that.__proto__ = Buffer.prototype;
  } else {
    // Fallback: Return an object instance of the Buffer class
    that = fromArrayLike(that, array);
  }
  return that
}

function fromObject (that, obj) {
  if (internalIsBuffer(obj)) {
    var len = checked(obj.length) | 0;
    that = createBuffer(that, len);

    if (that.length === 0) {
      return that
    }

    obj.copy(that, 0, 0, len);
    return that
  }

  if (obj) {
    if ((typeof ArrayBuffer !== 'undefined' &&
        obj.buffer instanceof ArrayBuffer) || 'length' in obj) {
      if (typeof obj.length !== 'number' || isnan(obj.length)) {
        return createBuffer(that, 0)
      }
      return fromArrayLike(that, obj)
    }

    if (obj.type === 'Buffer' && isArray(obj.data)) {
      return fromArrayLike(that, obj.data)
    }
  }

  throw new TypeError('First argument must be a string, Buffer, ArrayBuffer, Array, or array-like object.')
}

function checked (length) {
  // Note: cannot use `length < kMaxLength()` here because that fails when
  // length is NaN (which is otherwise coerced to zero.)
  if (length >= kMaxLength()) {
    throw new RangeError('Attempt to allocate Buffer larger than maximum ' +
                         'size: 0x' + kMaxLength().toString(16) + ' bytes')
  }
  return length | 0
}
Buffer.isBuffer = isBuffer;
function internalIsBuffer (b) {
  return !!(b != null && b._isBuffer)
}

Buffer.compare = function compare (a, b) {
  if (!internalIsBuffer(a) || !internalIsBuffer(b)) {
    throw new TypeError('Arguments must be Buffers')
  }

  if (a === b) return 0

  var x = a.length;
  var y = b.length;

  for (var i = 0, len = Math.min(x, y); i < len; ++i) {
    if (a[i] !== b[i]) {
      x = a[i];
      y = b[i];
      break
    }
  }

  if (x < y) return -1
  if (y < x) return 1
  return 0
};

Buffer.isEncoding = function isEncoding (encoding) {
  switch (String(encoding).toLowerCase()) {
    case 'hex':
    case 'utf8':
    case 'utf-8':
    case 'ascii':
    case 'latin1':
    case 'binary':
    case 'base64':
    case 'ucs2':
    case 'ucs-2':
    case 'utf16le':
    case 'utf-16le':
      return true
    default:
      return false
  }
};

Buffer.concat = function concat (list, length) {
  if (!isArray(list)) {
    throw new TypeError('"list" argument must be an Array of Buffers')
  }

  if (list.length === 0) {
    return Buffer.alloc(0)
  }

  var i;
  if (length === undefined) {
    length = 0;
    for (i = 0; i < list.length; ++i) {
      length += list[i].length;
    }
  }

  var buffer = Buffer.allocUnsafe(length);
  var pos = 0;
  for (i = 0; i < list.length; ++i) {
    var buf = list[i];
    if (!internalIsBuffer(buf)) {
      throw new TypeError('"list" argument must be an Array of Buffers')
    }
    buf.copy(buffer, pos);
    pos += buf.length;
  }
  return buffer
};

function byteLength (string, encoding) {
  if (internalIsBuffer(string)) {
    return string.length
  }
  if (typeof ArrayBuffer !== 'undefined' && typeof ArrayBuffer.isView === 'function' &&
      (ArrayBuffer.isView(string) || string instanceof ArrayBuffer)) {
    return string.byteLength
  }
  if (typeof string !== 'string') {
    string = '' + string;
  }

  var len = string.length;
  if (len === 0) return 0

  // Use a for loop to avoid recursion
  var loweredCase = false;
  for (;;) {
    switch (encoding) {
      case 'ascii':
      case 'latin1':
      case 'binary':
        return len
      case 'utf8':
      case 'utf-8':
      case undefined:
        return utf8ToBytes(string).length
      case 'ucs2':
      case 'ucs-2':
      case 'utf16le':
      case 'utf-16le':
        return len * 2
      case 'hex':
        return len >>> 1
      case 'base64':
        return base64ToBytes(string).length
      default:
        if (loweredCase) return utf8ToBytes(string).length // assume utf8
        encoding = ('' + encoding).toLowerCase();
        loweredCase = true;
    }
  }
}
Buffer.byteLength = byteLength;

function slowToString (encoding, start, end) {
  var loweredCase = false;

  // No need to verify that "this.length <= MAX_UINT32" since it's a read-only
  // property of a typed array.

  // This behaves neither like String nor Uint8Array in that we set start/end
  // to their upper/lower bounds if the value passed is out of range.
  // undefined is handled specially as per ECMA-262 6th Edition,
  // Section 13.3.3.7 Runtime Semantics: KeyedBindingInitialization.
  if (start === undefined || start < 0) {
    start = 0;
  }
  // Return early if start > this.length. Done here to prevent potential uint32
  // coercion fail below.
  if (start > this.length) {
    return ''
  }

  if (end === undefined || end > this.length) {
    end = this.length;
  }

  if (end <= 0) {
    return ''
  }

  // Force coersion to uint32. This will also coerce falsey/NaN values to 0.
  end >>>= 0;
  start >>>= 0;

  if (end <= start) {
    return ''
  }

  if (!encoding) encoding = 'utf8';

  while (true) {
    switch (encoding) {
      case 'hex':
        return hexSlice(this, start, end)

      case 'utf8':
      case 'utf-8':
        return utf8Slice(this, start, end)

      case 'ascii':
        return asciiSlice(this, start, end)

      case 'latin1':
      case 'binary':
        return latin1Slice(this, start, end)

      case 'base64':
        return base64Slice(this, start, end)

      case 'ucs2':
      case 'ucs-2':
      case 'utf16le':
      case 'utf-16le':
        return utf16leSlice(this, start, end)

      default:
        if (loweredCase) throw new TypeError('Unknown encoding: ' + encoding)
        encoding = (encoding + '').toLowerCase();
        loweredCase = true;
    }
  }
}

// The property is used by `Buffer.isBuffer` and `is-buffer` (in Safari 5-7) to detect
// Buffer instances.
Buffer.prototype._isBuffer = true;

function swap (b, n, m) {
  var i = b[n];
  b[n] = b[m];
  b[m] = i;
}

Buffer.prototype.swap16 = function swap16 () {
  var len = this.length;
  if (len % 2 !== 0) {
    throw new RangeError('Buffer size must be a multiple of 16-bits')
  }
  for (var i = 0; i < len; i += 2) {
    swap(this, i, i + 1);
  }
  return this
};

Buffer.prototype.swap32 = function swap32 () {
  var len = this.length;
  if (len % 4 !== 0) {
    throw new RangeError('Buffer size must be a multiple of 32-bits')
  }
  for (var i = 0; i < len; i += 4) {
    swap(this, i, i + 3);
    swap(this, i + 1, i + 2);
  }
  return this
};

Buffer.prototype.swap64 = function swap64 () {
  var len = this.length;
  if (len % 8 !== 0) {
    throw new RangeError('Buffer size must be a multiple of 64-bits')
  }
  for (var i = 0; i < len; i += 8) {
    swap(this, i, i + 7);
    swap(this, i + 1, i + 6);
    swap(this, i + 2, i + 5);
    swap(this, i + 3, i + 4);
  }
  return this
};

Buffer.prototype.toString = function toString () {
  var length = this.length | 0;
  if (length === 0) return ''
  if (arguments.length === 0) return utf8Slice(this, 0, length)
  return slowToString.apply(this, arguments)
};

Buffer.prototype.equals = function equals (b) {
  if (!internalIsBuffer(b)) throw new TypeError('Argument must be a Buffer')
  if (this === b) return true
  return Buffer.compare(this, b) === 0
};

Buffer.prototype.inspect = function inspect () {
  var str = '';
  var max = INSPECT_MAX_BYTES;
  if (this.length > 0) {
    str = this.toString('hex', 0, max).match(/.{2}/g).join(' ');
    if (this.length > max) str += ' ... ';
  }
  return '<Buffer ' + str + '>'
};

Buffer.prototype.compare = function compare (target, start, end, thisStart, thisEnd) {
  if (!internalIsBuffer(target)) {
    throw new TypeError('Argument must be a Buffer')
  }

  if (start === undefined) {
    start = 0;
  }
  if (end === undefined) {
    end = target ? target.length : 0;
  }
  if (thisStart === undefined) {
    thisStart = 0;
  }
  if (thisEnd === undefined) {
    thisEnd = this.length;
  }

  if (start < 0 || end > target.length || thisStart < 0 || thisEnd > this.length) {
    throw new RangeError('out of range index')
  }

  if (thisStart >= thisEnd && start >= end) {
    return 0
  }
  if (thisStart >= thisEnd) {
    return -1
  }
  if (start >= end) {
    return 1
  }

  start >>>= 0;
  end >>>= 0;
  thisStart >>>= 0;
  thisEnd >>>= 0;

  if (this === target) return 0

  var x = thisEnd - thisStart;
  var y = end - start;
  var len = Math.min(x, y);

  var thisCopy = this.slice(thisStart, thisEnd);
  var targetCopy = target.slice(start, end);

  for (var i = 0; i < len; ++i) {
    if (thisCopy[i] !== targetCopy[i]) {
      x = thisCopy[i];
      y = targetCopy[i];
      break
    }
  }

  if (x < y) return -1
  if (y < x) return 1
  return 0
};

// Finds either the first index of `val` in `buffer` at offset >= `byteOffset`,
// OR the last index of `val` in `buffer` at offset <= `byteOffset`.
//
// Arguments:
// - buffer - a Buffer to search
// - val - a string, Buffer, or number
// - byteOffset - an index into `buffer`; will be clamped to an int32
// - encoding - an optional encoding, relevant is val is a string
// - dir - true for indexOf, false for lastIndexOf
function bidirectionalIndexOf (buffer, val, byteOffset, encoding, dir) {
  // Empty buffer means no match
  if (buffer.length === 0) return -1

  // Normalize byteOffset
  if (typeof byteOffset === 'string') {
    encoding = byteOffset;
    byteOffset = 0;
  } else if (byteOffset > 0x7fffffff) {
    byteOffset = 0x7fffffff;
  } else if (byteOffset < -0x80000000) {
    byteOffset = -0x80000000;
  }
  byteOffset = +byteOffset;  // Coerce to Number.
  if (isNaN(byteOffset)) {
    // byteOffset: it it's undefined, null, NaN, "foo", etc, search whole buffer
    byteOffset = dir ? 0 : (buffer.length - 1);
  }

  // Normalize byteOffset: negative offsets start from the end of the buffer
  if (byteOffset < 0) byteOffset = buffer.length + byteOffset;
  if (byteOffset >= buffer.length) {
    if (dir) return -1
    else byteOffset = buffer.length - 1;
  } else if (byteOffset < 0) {
    if (dir) byteOffset = 0;
    else return -1
  }

  // Normalize val
  if (typeof val === 'string') {
    val = Buffer.from(val, encoding);
  }

  // Finally, search either indexOf (if dir is true) or lastIndexOf
  if (internalIsBuffer(val)) {
    // Special case: looking for empty string/buffer always fails
    if (val.length === 0) {
      return -1
    }
    return arrayIndexOf(buffer, val, byteOffset, encoding, dir)
  } else if (typeof val === 'number') {
    val = val & 0xFF; // Search for a byte value [0-255]
    if (Buffer.TYPED_ARRAY_SUPPORT &&
        typeof Uint8Array.prototype.indexOf === 'function') {
      if (dir) {
        return Uint8Array.prototype.indexOf.call(buffer, val, byteOffset)
      } else {
        return Uint8Array.prototype.lastIndexOf.call(buffer, val, byteOffset)
      }
    }
    return arrayIndexOf(buffer, [ val ], byteOffset, encoding, dir)
  }

  throw new TypeError('val must be string, number or Buffer')
}

function arrayIndexOf (arr, val, byteOffset, encoding, dir) {
  var indexSize = 1;
  var arrLength = arr.length;
  var valLength = val.length;

  if (encoding !== undefined) {
    encoding = String(encoding).toLowerCase();
    if (encoding === 'ucs2' || encoding === 'ucs-2' ||
        encoding === 'utf16le' || encoding === 'utf-16le') {
      if (arr.length < 2 || val.length < 2) {
        return -1
      }
      indexSize = 2;
      arrLength /= 2;
      valLength /= 2;
      byteOffset /= 2;
    }
  }

  function read (buf, i) {
    if (indexSize === 1) {
      return buf[i]
    } else {
      return buf.readUInt16BE(i * indexSize)
    }
  }

  var i;
  if (dir) {
    var foundIndex = -1;
    for (i = byteOffset; i < arrLength; i++) {
      if (read(arr, i) === read(val, foundIndex === -1 ? 0 : i - foundIndex)) {
        if (foundIndex === -1) foundIndex = i;
        if (i - foundIndex + 1 === valLength) return foundIndex * indexSize
      } else {
        if (foundIndex !== -1) i -= i - foundIndex;
        foundIndex = -1;
      }
    }
  } else {
    if (byteOffset + valLength > arrLength) byteOffset = arrLength - valLength;
    for (i = byteOffset; i >= 0; i--) {
      var found = true;
      for (var j = 0; j < valLength; j++) {
        if (read(arr, i + j) !== read(val, j)) {
          found = false;
          break
        }
      }
      if (found) return i
    }
  }

  return -1
}

Buffer.prototype.includes = function includes (val, byteOffset, encoding) {
  return this.indexOf(val, byteOffset, encoding) !== -1
};

Buffer.prototype.indexOf = function indexOf (val, byteOffset, encoding) {
  return bidirectionalIndexOf(this, val, byteOffset, encoding, true)
};

Buffer.prototype.lastIndexOf = function lastIndexOf (val, byteOffset, encoding) {
  return bidirectionalIndexOf(this, val, byteOffset, encoding, false)
};

function hexWrite (buf, string, offset, length) {
  offset = Number(offset) || 0;
  var remaining = buf.length - offset;
  if (!length) {
    length = remaining;
  } else {
    length = Number(length);
    if (length > remaining) {
      length = remaining;
    }
  }

  // must be an even number of digits
  var strLen = string.length;
  if (strLen % 2 !== 0) throw new TypeError('Invalid hex string')

  if (length > strLen / 2) {
    length = strLen / 2;
  }
  for (var i = 0; i < length; ++i) {
    var parsed = parseInt(string.substr(i * 2, 2), 16);
    if (isNaN(parsed)) return i
    buf[offset + i] = parsed;
  }
  return i
}

function utf8Write (buf, string, offset, length) {
  return blitBuffer(utf8ToBytes(string, buf.length - offset), buf, offset, length)
}

function asciiWrite (buf, string, offset, length) {
  return blitBuffer(asciiToBytes(string), buf, offset, length)
}

function latin1Write (buf, string, offset, length) {
  return asciiWrite(buf, string, offset, length)
}

function base64Write (buf, string, offset, length) {
  return blitBuffer(base64ToBytes(string), buf, offset, length)
}

function ucs2Write (buf, string, offset, length) {
  return blitBuffer(utf16leToBytes(string, buf.length - offset), buf, offset, length)
}

Buffer.prototype.write = function write (string, offset, length, encoding) {
  // Buffer#write(string)
  if (offset === undefined) {
    encoding = 'utf8';
    length = this.length;
    offset = 0;
  // Buffer#write(string, encoding)
  } else if (length === undefined && typeof offset === 'string') {
    encoding = offset;
    length = this.length;
    offset = 0;
  // Buffer#write(string, offset[, length][, encoding])
  } else if (isFinite(offset)) {
    offset = offset | 0;
    if (isFinite(length)) {
      length = length | 0;
      if (encoding === undefined) encoding = 'utf8';
    } else {
      encoding = length;
      length = undefined;
    }
  // legacy write(string, encoding, offset, length) - remove in v0.13
  } else {
    throw new Error(
      'Buffer.write(string, encoding, offset[, length]) is no longer supported'
    )
  }

  var remaining = this.length - offset;
  if (length === undefined || length > remaining) length = remaining;

  if ((string.length > 0 && (length < 0 || offset < 0)) || offset > this.length) {
    throw new RangeError('Attempt to write outside buffer bounds')
  }

  if (!encoding) encoding = 'utf8';

  var loweredCase = false;
  for (;;) {
    switch (encoding) {
      case 'hex':
        return hexWrite(this, string, offset, length)

      case 'utf8':
      case 'utf-8':
        return utf8Write(this, string, offset, length)

      case 'ascii':
        return asciiWrite(this, string, offset, length)

      case 'latin1':
      case 'binary':
        return latin1Write(this, string, offset, length)

      case 'base64':
        // Warning: maxLength not taken into account in base64Write
        return base64Write(this, string, offset, length)

      case 'ucs2':
      case 'ucs-2':
      case 'utf16le':
      case 'utf-16le':
        return ucs2Write(this, string, offset, length)

      default:
        if (loweredCase) throw new TypeError('Unknown encoding: ' + encoding)
        encoding = ('' + encoding).toLowerCase();
        loweredCase = true;
    }
  }
};

Buffer.prototype.toJSON = function toJSON () {
  return {
    type: 'Buffer',
    data: Array.prototype.slice.call(this._arr || this, 0)
  }
};

function base64Slice (buf, start, end) {
  if (start === 0 && end === buf.length) {
    return fromByteArray(buf)
  } else {
    return fromByteArray(buf.slice(start, end))
  }
}

function utf8Slice (buf, start, end) {
  end = Math.min(buf.length, end);
  var res = [];

  var i = start;
  while (i < end) {
    var firstByte = buf[i];
    var codePoint = null;
    var bytesPerSequence = (firstByte > 0xEF) ? 4
      : (firstByte > 0xDF) ? 3
      : (firstByte > 0xBF) ? 2
      : 1;

    if (i + bytesPerSequence <= end) {
      var secondByte, thirdByte, fourthByte, tempCodePoint;

      switch (bytesPerSequence) {
        case 1:
          if (firstByte < 0x80) {
            codePoint = firstByte;
          }
          break
        case 2:
          secondByte = buf[i + 1];
          if ((secondByte & 0xC0) === 0x80) {
            tempCodePoint = (firstByte & 0x1F) << 0x6 | (secondByte & 0x3F);
            if (tempCodePoint > 0x7F) {
              codePoint = tempCodePoint;
            }
          }
          break
        case 3:
          secondByte = buf[i + 1];
          thirdByte = buf[i + 2];
          if ((secondByte & 0xC0) === 0x80 && (thirdByte & 0xC0) === 0x80) {
            tempCodePoint = (firstByte & 0xF) << 0xC | (secondByte & 0x3F) << 0x6 | (thirdByte & 0x3F);
            if (tempCodePoint > 0x7FF && (tempCodePoint < 0xD800 || tempCodePoint > 0xDFFF)) {
              codePoint = tempCodePoint;
            }
          }
          break
        case 4:
          secondByte = buf[i + 1];
          thirdByte = buf[i + 2];
          fourthByte = buf[i + 3];
          if ((secondByte & 0xC0) === 0x80 && (thirdByte & 0xC0) === 0x80 && (fourthByte & 0xC0) === 0x80) {
            tempCodePoint = (firstByte & 0xF) << 0x12 | (secondByte & 0x3F) << 0xC | (thirdByte & 0x3F) << 0x6 | (fourthByte & 0x3F);
            if (tempCodePoint > 0xFFFF && tempCodePoint < 0x110000) {
              codePoint = tempCodePoint;
            }
          }
      }
    }

    if (codePoint === null) {
      // we did not generate a valid codePoint so insert a
      // replacement char (U+FFFD) and advance only 1 byte
      codePoint = 0xFFFD;
      bytesPerSequence = 1;
    } else if (codePoint > 0xFFFF) {
      // encode to utf16 (surrogate pair dance)
      codePoint -= 0x10000;
      res.push(codePoint >>> 10 & 0x3FF | 0xD800);
      codePoint = 0xDC00 | codePoint & 0x3FF;
    }

    res.push(codePoint);
    i += bytesPerSequence;
  }

  return decodeCodePointsArray(res)
}

// Based on http://stackoverflow.com/a/22747272/680742, the browser with
// the lowest limit is Chrome, with 0x10000 args.
// We go 1 magnitude less, for safety
var MAX_ARGUMENTS_LENGTH = 0x1000;

function decodeCodePointsArray (codePoints) {
  var len = codePoints.length;
  if (len <= MAX_ARGUMENTS_LENGTH) {
    return String.fromCharCode.apply(String, codePoints) // avoid extra slice()
  }

  // Decode in chunks to avoid "call stack size exceeded".
  var res = '';
  var i = 0;
  while (i < len) {
    res += String.fromCharCode.apply(
      String,
      codePoints.slice(i, i += MAX_ARGUMENTS_LENGTH)
    );
  }
  return res
}

function asciiSlice (buf, start, end) {
  var ret = '';
  end = Math.min(buf.length, end);

  for (var i = start; i < end; ++i) {
    ret += String.fromCharCode(buf[i] & 0x7F);
  }
  return ret
}

function latin1Slice (buf, start, end) {
  var ret = '';
  end = Math.min(buf.length, end);

  for (var i = start; i < end; ++i) {
    ret += String.fromCharCode(buf[i]);
  }
  return ret
}

function hexSlice (buf, start, end) {
  var len = buf.length;

  if (!start || start < 0) start = 0;
  if (!end || end < 0 || end > len) end = len;

  var out = '';
  for (var i = start; i < end; ++i) {
    out += toHex(buf[i]);
  }
  return out
}

function utf16leSlice (buf, start, end) {
  var bytes = buf.slice(start, end);
  var res = '';
  for (var i = 0; i < bytes.length; i += 2) {
    res += String.fromCharCode(bytes[i] + bytes[i + 1] * 256);
  }
  return res
}

Buffer.prototype.slice = function slice (start, end) {
  var len = this.length;
  start = ~~start;
  end = end === undefined ? len : ~~end;

  if (start < 0) {
    start += len;
    if (start < 0) start = 0;
  } else if (start > len) {
    start = len;
  }

  if (end < 0) {
    end += len;
    if (end < 0) end = 0;
  } else if (end > len) {
    end = len;
  }

  if (end < start) end = start;

  var newBuf;
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    newBuf = this.subarray(start, end);
    newBuf.__proto__ = Buffer.prototype;
  } else {
    var sliceLen = end - start;
    newBuf = new Buffer(sliceLen, undefined);
    for (var i = 0; i < sliceLen; ++i) {
      newBuf[i] = this[i + start];
    }
  }

  return newBuf
};

/*
 * Need to make sure that buffer isn't trying to write out of bounds.
 */
function checkOffset (offset, ext, length) {
  if ((offset % 1) !== 0 || offset < 0) throw new RangeError('offset is not uint')
  if (offset + ext > length) throw new RangeError('Trying to access beyond buffer length')
}

Buffer.prototype.readUIntLE = function readUIntLE (offset, byteLength, noAssert) {
  offset = offset | 0;
  byteLength = byteLength | 0;
  if (!noAssert) checkOffset(offset, byteLength, this.length);

  var val = this[offset];
  var mul = 1;
  var i = 0;
  while (++i < byteLength && (mul *= 0x100)) {
    val += this[offset + i] * mul;
  }

  return val
};

Buffer.prototype.readUIntBE = function readUIntBE (offset, byteLength, noAssert) {
  offset = offset | 0;
  byteLength = byteLength | 0;
  if (!noAssert) {
    checkOffset(offset, byteLength, this.length);
  }

  var val = this[offset + --byteLength];
  var mul = 1;
  while (byteLength > 0 && (mul *= 0x100)) {
    val += this[offset + --byteLength] * mul;
  }

  return val
};

Buffer.prototype.readUInt8 = function readUInt8 (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 1, this.length);
  return this[offset]
};

Buffer.prototype.readUInt16LE = function readUInt16LE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 2, this.length);
  return this[offset] | (this[offset + 1] << 8)
};

Buffer.prototype.readUInt16BE = function readUInt16BE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 2, this.length);
  return (this[offset] << 8) | this[offset + 1]
};

Buffer.prototype.readUInt32LE = function readUInt32LE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 4, this.length);

  return ((this[offset]) |
      (this[offset + 1] << 8) |
      (this[offset + 2] << 16)) +
      (this[offset + 3] * 0x1000000)
};

Buffer.prototype.readUInt32BE = function readUInt32BE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 4, this.length);

  return (this[offset] * 0x1000000) +
    ((this[offset + 1] << 16) |
    (this[offset + 2] << 8) |
    this[offset + 3])
};

Buffer.prototype.readIntLE = function readIntLE (offset, byteLength, noAssert) {
  offset = offset | 0;
  byteLength = byteLength | 0;
  if (!noAssert) checkOffset(offset, byteLength, this.length);

  var val = this[offset];
  var mul = 1;
  var i = 0;
  while (++i < byteLength && (mul *= 0x100)) {
    val += this[offset + i] * mul;
  }
  mul *= 0x80;

  if (val >= mul) val -= Math.pow(2, 8 * byteLength);

  return val
};

Buffer.prototype.readIntBE = function readIntBE (offset, byteLength, noAssert) {
  offset = offset | 0;
  byteLength = byteLength | 0;
  if (!noAssert) checkOffset(offset, byteLength, this.length);

  var i = byteLength;
  var mul = 1;
  var val = this[offset + --i];
  while (i > 0 && (mul *= 0x100)) {
    val += this[offset + --i] * mul;
  }
  mul *= 0x80;

  if (val >= mul) val -= Math.pow(2, 8 * byteLength);

  return val
};

Buffer.prototype.readInt8 = function readInt8 (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 1, this.length);
  if (!(this[offset] & 0x80)) return (this[offset])
  return ((0xff - this[offset] + 1) * -1)
};

Buffer.prototype.readInt16LE = function readInt16LE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 2, this.length);
  var val = this[offset] | (this[offset + 1] << 8);
  return (val & 0x8000) ? val | 0xFFFF0000 : val
};

Buffer.prototype.readInt16BE = function readInt16BE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 2, this.length);
  var val = this[offset + 1] | (this[offset] << 8);
  return (val & 0x8000) ? val | 0xFFFF0000 : val
};

Buffer.prototype.readInt32LE = function readInt32LE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 4, this.length);

  return (this[offset]) |
    (this[offset + 1] << 8) |
    (this[offset + 2] << 16) |
    (this[offset + 3] << 24)
};

Buffer.prototype.readInt32BE = function readInt32BE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 4, this.length);

  return (this[offset] << 24) |
    (this[offset + 1] << 16) |
    (this[offset + 2] << 8) |
    (this[offset + 3])
};

Buffer.prototype.readFloatLE = function readFloatLE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 4, this.length);
  return read(this, offset, true, 23, 4)
};

Buffer.prototype.readFloatBE = function readFloatBE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 4, this.length);
  return read(this, offset, false, 23, 4)
};

Buffer.prototype.readDoubleLE = function readDoubleLE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 8, this.length);
  return read(this, offset, true, 52, 8)
};

Buffer.prototype.readDoubleBE = function readDoubleBE (offset, noAssert) {
  if (!noAssert) checkOffset(offset, 8, this.length);
  return read(this, offset, false, 52, 8)
};

function checkInt (buf, value, offset, ext, max, min) {
  if (!internalIsBuffer(buf)) throw new TypeError('"buffer" argument must be a Buffer instance')
  if (value > max || value < min) throw new RangeError('"value" argument is out of bounds')
  if (offset + ext > buf.length) throw new RangeError('Index out of range')
}

Buffer.prototype.writeUIntLE = function writeUIntLE (value, offset, byteLength, noAssert) {
  value = +value;
  offset = offset | 0;
  byteLength = byteLength | 0;
  if (!noAssert) {
    var maxBytes = Math.pow(2, 8 * byteLength) - 1;
    checkInt(this, value, offset, byteLength, maxBytes, 0);
  }

  var mul = 1;
  var i = 0;
  this[offset] = value & 0xFF;
  while (++i < byteLength && (mul *= 0x100)) {
    this[offset + i] = (value / mul) & 0xFF;
  }

  return offset + byteLength
};

Buffer.prototype.writeUIntBE = function writeUIntBE (value, offset, byteLength, noAssert) {
  value = +value;
  offset = offset | 0;
  byteLength = byteLength | 0;
  if (!noAssert) {
    var maxBytes = Math.pow(2, 8 * byteLength) - 1;
    checkInt(this, value, offset, byteLength, maxBytes, 0);
  }

  var i = byteLength - 1;
  var mul = 1;
  this[offset + i] = value & 0xFF;
  while (--i >= 0 && (mul *= 0x100)) {
    this[offset + i] = (value / mul) & 0xFF;
  }

  return offset + byteLength
};

Buffer.prototype.writeUInt8 = function writeUInt8 (value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert) checkInt(this, value, offset, 1, 0xff, 0);
  if (!Buffer.TYPED_ARRAY_SUPPORT) value = Math.floor(value);
  this[offset] = (value & 0xff);
  return offset + 1
};

function objectWriteUInt16 (buf, value, offset, littleEndian) {
  if (value < 0) value = 0xffff + value + 1;
  for (var i = 0, j = Math.min(buf.length - offset, 2); i < j; ++i) {
    buf[offset + i] = (value & (0xff << (8 * (littleEndian ? i : 1 - i)))) >>>
      (littleEndian ? i : 1 - i) * 8;
  }
}

Buffer.prototype.writeUInt16LE = function writeUInt16LE (value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert) checkInt(this, value, offset, 2, 0xffff, 0);
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = (value & 0xff);
    this[offset + 1] = (value >>> 8);
  } else {
    objectWriteUInt16(this, value, offset, true);
  }
  return offset + 2
};

Buffer.prototype.writeUInt16BE = function writeUInt16BE (value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert) checkInt(this, value, offset, 2, 0xffff, 0);
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = (value >>> 8);
    this[offset + 1] = (value & 0xff);
  } else {
    objectWriteUInt16(this, value, offset, false);
  }
  return offset + 2
};

function objectWriteUInt32 (buf, value, offset, littleEndian) {
  if (value < 0) value = 0xffffffff + value + 1;
  for (var i = 0, j = Math.min(buf.length - offset, 4); i < j; ++i) {
    buf[offset + i] = (value >>> (littleEndian ? i : 3 - i) * 8) & 0xff;
  }
}

Buffer.prototype.writeUInt32LE = function writeUInt32LE (value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert) checkInt(this, value, offset, 4, 0xffffffff, 0);
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset + 3] = (value >>> 24);
    this[offset + 2] = (value >>> 16);
    this[offset + 1] = (value >>> 8);
    this[offset] = (value & 0xff);
  } else {
    objectWriteUInt32(this, value, offset, true);
  }
  return offset + 4
};

Buffer.prototype.writeUInt32BE = function writeUInt32BE (value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert) checkInt(this, value, offset, 4, 0xffffffff, 0);
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = (value >>> 24);
    this[offset + 1] = (value >>> 16);
    this[offset + 2] = (value >>> 8);
    this[offset + 3] = (value & 0xff);
  } else {
    objectWriteUInt32(this, value, offset, false);
  }
  return offset + 4
};

Buffer.prototype.writeIntLE = function writeIntLE (value, offset, byteLength, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert) {
    var limit = Math.pow(2, 8 * byteLength - 1);

    checkInt(this, value, offset, byteLength, limit - 1, -limit);
  }

  var i = 0;
  var mul = 1;
  var sub = 0;
  this[offset] = value & 0xFF;
  while (++i < byteLength && (mul *= 0x100)) {
    if (value < 0 && sub === 0 && this[offset + i - 1] !== 0) {
      sub = 1;
    }
    this[offset + i] = ((value / mul) >> 0) - sub & 0xFF;
  }

  return offset + byteLength
};

Buffer.prototype.writeIntBE = function writeIntBE (value, offset, byteLength, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert) {
    var limit = Math.pow(2, 8 * byteLength - 1);

    checkInt(this, value, offset, byteLength, limit - 1, -limit);
  }

  var i = byteLength - 1;
  var mul = 1;
  var sub = 0;
  this[offset + i] = value & 0xFF;
  while (--i >= 0 && (mul *= 0x100)) {
    if (value < 0 && sub === 0 && this[offset + i + 1] !== 0) {
      sub = 1;
    }
    this[offset + i] = ((value / mul) >> 0) - sub & 0xFF;
  }

  return offset + byteLength
};

Buffer.prototype.writeInt8 = function writeInt8 (value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert) checkInt(this, value, offset, 1, 0x7f, -0x80);
  if (!Buffer.TYPED_ARRAY_SUPPORT) value = Math.floor(value);
  if (value < 0) value = 0xff + value + 1;
  this[offset] = (value & 0xff);
  return offset + 1
};

Buffer.prototype.writeInt16LE = function writeInt16LE (value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert) checkInt(this, value, offset, 2, 0x7fff, -0x8000);
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = (value & 0xff);
    this[offset + 1] = (value >>> 8);
  } else {
    objectWriteUInt16(this, value, offset, true);
  }
  return offset + 2
};

Buffer.prototype.writeInt16BE = function writeInt16BE (value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert) checkInt(this, value, offset, 2, 0x7fff, -0x8000);
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = (value >>> 8);
    this[offset + 1] = (value & 0xff);
  } else {
    objectWriteUInt16(this, value, offset, false);
  }
  return offset + 2
};

Buffer.prototype.writeInt32LE = function writeInt32LE (value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert) checkInt(this, value, offset, 4, 0x7fffffff, -0x80000000);
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = (value & 0xff);
    this[offset + 1] = (value >>> 8);
    this[offset + 2] = (value >>> 16);
    this[offset + 3] = (value >>> 24);
  } else {
    objectWriteUInt32(this, value, offset, true);
  }
  return offset + 4
};

Buffer.prototype.writeInt32BE = function writeInt32BE (value, offset, noAssert) {
  value = +value;
  offset = offset | 0;
  if (!noAssert) checkInt(this, value, offset, 4, 0x7fffffff, -0x80000000);
  if (value < 0) value = 0xffffffff + value + 1;
  if (Buffer.TYPED_ARRAY_SUPPORT) {
    this[offset] = (value >>> 24);
    this[offset + 1] = (value >>> 16);
    this[offset + 2] = (value >>> 8);
    this[offset + 3] = (value & 0xff);
  } else {
    objectWriteUInt32(this, value, offset, false);
  }
  return offset + 4
};

function checkIEEE754 (buf, value, offset, ext, max, min) {
  if (offset + ext > buf.length) throw new RangeError('Index out of range')
  if (offset < 0) throw new RangeError('Index out of range')
}

function writeFloat (buf, value, offset, littleEndian, noAssert) {
  if (!noAssert) {
    checkIEEE754(buf, value, offset, 4);
  }
  write(buf, value, offset, littleEndian, 23, 4);
  return offset + 4
}

Buffer.prototype.writeFloatLE = function writeFloatLE (value, offset, noAssert) {
  return writeFloat(this, value, offset, true, noAssert)
};

Buffer.prototype.writeFloatBE = function writeFloatBE (value, offset, noAssert) {
  return writeFloat(this, value, offset, false, noAssert)
};

function writeDouble (buf, value, offset, littleEndian, noAssert) {
  if (!noAssert) {
    checkIEEE754(buf, value, offset, 8);
  }
  write(buf, value, offset, littleEndian, 52, 8);
  return offset + 8
}

Buffer.prototype.writeDoubleLE = function writeDoubleLE (value, offset, noAssert) {
  return writeDouble(this, value, offset, true, noAssert)
};

Buffer.prototype.writeDoubleBE = function writeDoubleBE (value, offset, noAssert) {
  return writeDouble(this, value, offset, false, noAssert)
};

// copy(targetBuffer, targetStart=0, sourceStart=0, sourceEnd=buffer.length)
Buffer.prototype.copy = function copy (target, targetStart, start, end) {
  if (!start) start = 0;
  if (!end && end !== 0) end = this.length;
  if (targetStart >= target.length) targetStart = target.length;
  if (!targetStart) targetStart = 0;
  if (end > 0 && end < start) end = start;

  // Copy 0 bytes; we're done
  if (end === start) return 0
  if (target.length === 0 || this.length === 0) return 0

  // Fatal error conditions
  if (targetStart < 0) {
    throw new RangeError('targetStart out of bounds')
  }
  if (start < 0 || start >= this.length) throw new RangeError('sourceStart out of bounds')
  if (end < 0) throw new RangeError('sourceEnd out of bounds')

  // Are we oob?
  if (end > this.length) end = this.length;
  if (target.length - targetStart < end - start) {
    end = target.length - targetStart + start;
  }

  var len = end - start;
  var i;

  if (this === target && start < targetStart && targetStart < end) {
    // descending copy from end
    for (i = len - 1; i >= 0; --i) {
      target[i + targetStart] = this[i + start];
    }
  } else if (len < 1000 || !Buffer.TYPED_ARRAY_SUPPORT) {
    // ascending copy from start
    for (i = 0; i < len; ++i) {
      target[i + targetStart] = this[i + start];
    }
  } else {
    Uint8Array.prototype.set.call(
      target,
      this.subarray(start, start + len),
      targetStart
    );
  }

  return len
};

// Usage:
//    buffer.fill(number[, offset[, end]])
//    buffer.fill(buffer[, offset[, end]])
//    buffer.fill(string[, offset[, end]][, encoding])
Buffer.prototype.fill = function fill (val, start, end, encoding) {
  // Handle string cases:
  if (typeof val === 'string') {
    if (typeof start === 'string') {
      encoding = start;
      start = 0;
      end = this.length;
    } else if (typeof end === 'string') {
      encoding = end;
      end = this.length;
    }
    if (val.length === 1) {
      var code = val.charCodeAt(0);
      if (code < 256) {
        val = code;
      }
    }
    if (encoding !== undefined && typeof encoding !== 'string') {
      throw new TypeError('encoding must be a string')
    }
    if (typeof encoding === 'string' && !Buffer.isEncoding(encoding)) {
      throw new TypeError('Unknown encoding: ' + encoding)
    }
  } else if (typeof val === 'number') {
    val = val & 255;
  }

  // Invalid ranges are not set to a default, so can range check early.
  if (start < 0 || this.length < start || this.length < end) {
    throw new RangeError('Out of range index')
  }

  if (end <= start) {
    return this
  }

  start = start >>> 0;
  end = end === undefined ? this.length : end >>> 0;

  if (!val) val = 0;

  var i;
  if (typeof val === 'number') {
    for (i = start; i < end; ++i) {
      this[i] = val;
    }
  } else {
    var bytes = internalIsBuffer(val)
      ? val
      : utf8ToBytes(new Buffer(val, encoding).toString());
    var len = bytes.length;
    for (i = 0; i < end - start; ++i) {
      this[i + start] = bytes[i % len];
    }
  }

  return this
};

// HELPER FUNCTIONS
// ================

var INVALID_BASE64_RE = /[^+\/0-9A-Za-z-_]/g;

function base64clean (str) {
  // Node strips out invalid characters like \n and \t from the string, base64-js does not
  str = stringtrim(str).replace(INVALID_BASE64_RE, '');
  // Node converts strings with length < 2 to ''
  if (str.length < 2) return ''
  // Node allows for non-padded base64 strings (missing trailing ===), base64-js does not
  while (str.length % 4 !== 0) {
    str = str + '=';
  }
  return str
}

function stringtrim (str) {
  if (str.trim) return str.trim()
  return str.replace(/^\s+|\s+$/g, '')
}

function toHex (n) {
  if (n < 16) return '0' + n.toString(16)
  return n.toString(16)
}

function utf8ToBytes (string, units) {
  units = units || Infinity;
  var codePoint;
  var length = string.length;
  var leadSurrogate = null;
  var bytes = [];

  for (var i = 0; i < length; ++i) {
    codePoint = string.charCodeAt(i);

    // is surrogate component
    if (codePoint > 0xD7FF && codePoint < 0xE000) {
      // last char was a lead
      if (!leadSurrogate) {
        // no lead yet
        if (codePoint > 0xDBFF) {
          // unexpected trail
          if ((units -= 3) > -1) bytes.push(0xEF, 0xBF, 0xBD);
          continue
        } else if (i + 1 === length) {
          // unpaired lead
          if ((units -= 3) > -1) bytes.push(0xEF, 0xBF, 0xBD);
          continue
        }

        // valid lead
        leadSurrogate = codePoint;

        continue
      }

      // 2 leads in a row
      if (codePoint < 0xDC00) {
        if ((units -= 3) > -1) bytes.push(0xEF, 0xBF, 0xBD);
        leadSurrogate = codePoint;
        continue
      }

      // valid surrogate pair
      codePoint = (leadSurrogate - 0xD800 << 10 | codePoint - 0xDC00) + 0x10000;
    } else if (leadSurrogate) {
      // valid bmp char, but last char was a lead
      if ((units -= 3) > -1) bytes.push(0xEF, 0xBF, 0xBD);
    }

    leadSurrogate = null;

    // encode utf8
    if (codePoint < 0x80) {
      if ((units -= 1) < 0) break
      bytes.push(codePoint);
    } else if (codePoint < 0x800) {
      if ((units -= 2) < 0) break
      bytes.push(
        codePoint >> 0x6 | 0xC0,
        codePoint & 0x3F | 0x80
      );
    } else if (codePoint < 0x10000) {
      if ((units -= 3) < 0) break
      bytes.push(
        codePoint >> 0xC | 0xE0,
        codePoint >> 0x6 & 0x3F | 0x80,
        codePoint & 0x3F | 0x80
      );
    } else if (codePoint < 0x110000) {
      if ((units -= 4) < 0) break
      bytes.push(
        codePoint >> 0x12 | 0xF0,
        codePoint >> 0xC & 0x3F | 0x80,
        codePoint >> 0x6 & 0x3F | 0x80,
        codePoint & 0x3F | 0x80
      );
    } else {
      throw new Error('Invalid code point')
    }
  }

  return bytes
}

function asciiToBytes (str) {
  var byteArray = [];
  for (var i = 0; i < str.length; ++i) {
    // Node's code seems to be doing this and not & 0x7F..
    byteArray.push(str.charCodeAt(i) & 0xFF);
  }
  return byteArray
}

function utf16leToBytes (str, units) {
  var c, hi, lo;
  var byteArray = [];
  for (var i = 0; i < str.length; ++i) {
    if ((units -= 2) < 0) break

    c = str.charCodeAt(i);
    hi = c >> 8;
    lo = c % 256;
    byteArray.push(lo);
    byteArray.push(hi);
  }

  return byteArray
}


function base64ToBytes (str) {
  return toByteArray(base64clean(str))
}

function blitBuffer (src, dst, offset, length) {
  for (var i = 0; i < length; ++i) {
    if ((i + offset >= dst.length) || (i >= src.length)) break
    dst[i + offset] = src[i];
  }
  return i
}

function isnan (val) {
  return val !== val // eslint-disable-line no-self-compare
}


// the following is from is-buffer, also by Feross Aboukhadijeh and with same lisence
// The _isBuffer check is for Safari 5-7 support, because it's missing
// Object.prototype.constructor. Remove this eventually
function isBuffer(obj) {
  return obj != null && (!!obj._isBuffer || isFastBuffer(obj) || isSlowBuffer(obj))
}

function isFastBuffer (obj) {
  return !!obj.constructor && typeof obj.constructor.isBuffer === 'function' && obj.constructor.isBuffer(obj)
}

// For Node v0.10 support. Remove this eventually.
function isSlowBuffer (obj) {
  return typeof obj.readFloatLE === 'function' && typeof obj.slice === 'function' && isFastBuffer(obj.slice(0, 0))
}

class Context {
    constructor(importPath, useContextVariables) {
        this.code = "";
        this.scopes = [["vars"]];
        this.bitFields = [];
        this.tmpVariableCount = 0;
        this.references = new Map();
        this.imports = [];
        this.reverseImports = new Map();
        this.useContextVariables = false;
        this.importPath = importPath;
        this.useContextVariables = useContextVariables;
    }
    generateVariable(name) {
        const scopes = [...this.scopes[this.scopes.length - 1]];
        if (name) {
            scopes.push(name);
        }
        return scopes.join(".");
    }
    generateOption(val) {
        switch (typeof val) {
            case "number":
                return val.toString();
            case "string":
                return this.generateVariable(val);
            case "function":
                return `${this.addImport(val)}.call(${this.generateVariable()}, vars)`;
        }
    }
    generateError(err) {
        this.pushCode(`throw new Error(${err});`);
    }
    generateTmpVariable() {
        return "$tmp" + this.tmpVariableCount++;
    }
    pushCode(code) {
        this.code += code + "\n";
    }
    pushPath(name) {
        if (name) {
            this.scopes[this.scopes.length - 1].push(name);
        }
    }
    popPath(name) {
        if (name) {
            this.scopes[this.scopes.length - 1].pop();
        }
    }
    pushScope(name) {
        this.scopes.push([name]);
    }
    popScope() {
        this.scopes.pop();
    }
    addImport(im) {
        if (!this.importPath)
            return `(${im})`;
        let id = this.reverseImports.get(im);
        if (!id) {
            id = this.imports.push(im) - 1;
            this.reverseImports.set(im, id);
        }
        return `${this.importPath}[${id}]`;
    }
    addReference(alias) {
        if (!this.references.has(alias)) {
            this.references.set(alias, { resolved: false, requested: false });
        }
    }
    markResolved(alias) {
        const reference = this.references.get(alias);
        if (reference) {
            reference.resolved = true;
        }
    }
    markRequested(aliasList) {
        aliasList.forEach((alias) => {
            const reference = this.references.get(alias);
            if (reference) {
                reference.requested = true;
            }
        });
    }
    getUnresolvedReferences() {
        return Array.from(this.references)
            .filter(([_, reference]) => !reference.resolved && !reference.requested)
            .map(([alias, _]) => alias);
    }
}
const aliasRegistry = new Map();
const FUNCTION_PREFIX = "___parser_";
const PRIMITIVE_SIZES = {
    uint8: 1,
    uint16le: 2,
    uint16be: 2,
    uint32le: 4,
    uint32be: 4,
    int8: 1,
    int16le: 2,
    int16be: 2,
    int32le: 4,
    int32be: 4,
    int64be: 8,
    int64le: 8,
    uint64be: 8,
    uint64le: 8,
    floatle: 4,
    floatbe: 4,
    doublele: 8,
    doublebe: 8,
};
const PRIMITIVE_NAMES = {
    uint8: "Uint8",
    uint16le: "Uint16",
    uint16be: "Uint16",
    uint32le: "Uint32",
    uint32be: "Uint32",
    int8: "Int8",
    int16le: "Int16",
    int16be: "Int16",
    int32le: "Int32",
    int32be: "Int32",
    int64be: "BigInt64",
    int64le: "BigInt64",
    uint64be: "BigUint64",
    uint64le: "BigUint64",
    floatle: "Float32",
    floatbe: "Float32",
    doublele: "Float64",
    doublebe: "Float64",
};
const PRIMITIVE_LITTLE_ENDIANS = {
    uint8: false,
    uint16le: true,
    uint16be: false,
    uint32le: true,
    uint32be: false,
    int8: false,
    int16le: true,
    int16be: false,
    int32le: true,
    int32be: false,
    int64be: false,
    int64le: true,
    uint64be: false,
    uint64le: true,
    floatle: true,
    floatbe: false,
    doublele: true,
    doublebe: false,
};
class Parser {
    constructor() {
        this.varName = "";
        this.type = "";
        this.options = {};
        this.endian = "be";
        this.useContextVariables = false;
    }
    static start() {
        return new Parser();
    }
    primitiveGenerateN(type, ctx) {
        const typeName = PRIMITIVE_NAMES[type];
        const littleEndian = PRIMITIVE_LITTLE_ENDIANS[type];
        ctx.pushCode(`${ctx.generateVariable(this.varName)} = dataView.get${typeName}(offset, ${littleEndian});`);
        ctx.pushCode(`offset += ${PRIMITIVE_SIZES[type]};`);
    }
    primitiveN(type, varName, options) {
        return this.setNextParser(type, varName, options);
    }
    useThisEndian(type) {
        return (type + this.endian.toLowerCase());
    }
    uint8(varName, options = {}) {
        return this.primitiveN("uint8", varName, options);
    }
    uint16(varName, options = {}) {
        return this.primitiveN(this.useThisEndian("uint16"), varName, options);
    }
    uint16le(varName, options = {}) {
        return this.primitiveN("uint16le", varName, options);
    }
    uint16be(varName, options = {}) {
        return this.primitiveN("uint16be", varName, options);
    }
    uint32(varName, options = {}) {
        return this.primitiveN(this.useThisEndian("uint32"), varName, options);
    }
    uint32le(varName, options = {}) {
        return this.primitiveN("uint32le", varName, options);
    }
    uint32be(varName, options = {}) {
        return this.primitiveN("uint32be", varName, options);
    }
    int8(varName, options = {}) {
        return this.primitiveN("int8", varName, options);
    }
    int16(varName, options = {}) {
        return this.primitiveN(this.useThisEndian("int16"), varName, options);
    }
    int16le(varName, options = {}) {
        return this.primitiveN("int16le", varName, options);
    }
    int16be(varName, options = {}) {
        return this.primitiveN("int16be", varName, options);
    }
    int32(varName, options = {}) {
        return this.primitiveN(this.useThisEndian("int32"), varName, options);
    }
    int32le(varName, options = {}) {
        return this.primitiveN("int32le", varName, options);
    }
    int32be(varName, options = {}) {
        return this.primitiveN("int32be", varName, options);
    }
    bigIntVersionCheck() {
        if (!DataView.prototype.getBigInt64)
            throw new Error("BigInt64 is unsupported on this runtime");
    }
    int64(varName, options = {}) {
        this.bigIntVersionCheck();
        return this.primitiveN(this.useThisEndian("int64"), varName, options);
    }
    int64be(varName, options = {}) {
        this.bigIntVersionCheck();
        return this.primitiveN("int64be", varName, options);
    }
    int64le(varName, options = {}) {
        this.bigIntVersionCheck();
        return this.primitiveN("int64le", varName, options);
    }
    uint64(varName, options = {}) {
        this.bigIntVersionCheck();
        return this.primitiveN(this.useThisEndian("uint64"), varName, options);
    }
    uint64be(varName, options = {}) {
        this.bigIntVersionCheck();
        return this.primitiveN("uint64be", varName, options);
    }
    uint64le(varName, options = {}) {
        this.bigIntVersionCheck();
        return this.primitiveN("uint64le", varName, options);
    }
    floatle(varName, options = {}) {
        return this.primitiveN("floatle", varName, options);
    }
    floatbe(varName, options = {}) {
        return this.primitiveN("floatbe", varName, options);
    }
    doublele(varName, options = {}) {
        return this.primitiveN("doublele", varName, options);
    }
    doublebe(varName, options = {}) {
        return this.primitiveN("doublebe", varName, options);
    }
    bitN(size, varName, options) {
        options.length = size;
        return this.setNextParser("bit", varName, options);
    }
    bit1(varName, options = {}) {
        return this.bitN(1, varName, options);
    }
    bit2(varName, options = {}) {
        return this.bitN(2, varName, options);
    }
    bit3(varName, options = {}) {
        return this.bitN(3, varName, options);
    }
    bit4(varName, options = {}) {
        return this.bitN(4, varName, options);
    }
    bit5(varName, options = {}) {
        return this.bitN(5, varName, options);
    }
    bit6(varName, options = {}) {
        return this.bitN(6, varName, options);
    }
    bit7(varName, options = {}) {
        return this.bitN(7, varName, options);
    }
    bit8(varName, options = {}) {
        return this.bitN(8, varName, options);
    }
    bit9(varName, options = {}) {
        return this.bitN(9, varName, options);
    }
    bit10(varName, options = {}) {
        return this.bitN(10, varName, options);
    }
    bit11(varName, options = {}) {
        return this.bitN(11, varName, options);
    }
    bit12(varName, options = {}) {
        return this.bitN(12, varName, options);
    }
    bit13(varName, options = {}) {
        return this.bitN(13, varName, options);
    }
    bit14(varName, options = {}) {
        return this.bitN(14, varName, options);
    }
    bit15(varName, options = {}) {
        return this.bitN(15, varName, options);
    }
    bit16(varName, options = {}) {
        return this.bitN(16, varName, options);
    }
    bit17(varName, options = {}) {
        return this.bitN(17, varName, options);
    }
    bit18(varName, options = {}) {
        return this.bitN(18, varName, options);
    }
    bit19(varName, options = {}) {
        return this.bitN(19, varName, options);
    }
    bit20(varName, options = {}) {
        return this.bitN(20, varName, options);
    }
    bit21(varName, options = {}) {
        return this.bitN(21, varName, options);
    }
    bit22(varName, options = {}) {
        return this.bitN(22, varName, options);
    }
    bit23(varName, options = {}) {
        return this.bitN(23, varName, options);
    }
    bit24(varName, options = {}) {
        return this.bitN(24, varName, options);
    }
    bit25(varName, options = {}) {
        return this.bitN(25, varName, options);
    }
    bit26(varName, options = {}) {
        return this.bitN(26, varName, options);
    }
    bit27(varName, options = {}) {
        return this.bitN(27, varName, options);
    }
    bit28(varName, options = {}) {
        return this.bitN(28, varName, options);
    }
    bit29(varName, options = {}) {
        return this.bitN(29, varName, options);
    }
    bit30(varName, options = {}) {
        return this.bitN(30, varName, options);
    }
    bit31(varName, options = {}) {
        return this.bitN(31, varName, options);
    }
    bit32(varName, options = {}) {
        return this.bitN(32, varName, options);
    }
    namely(alias) {
        aliasRegistry.set(alias, this);
        this.alias = alias;
        return this;
    }
    skip(length, options = {}) {
        return this.seek(length, options);
    }
    seek(relOffset, options = {}) {
        if (options.assert) {
            throw new Error("assert option on seek is not allowed.");
        }
        return this.setNextParser("seek", "", { length: relOffset });
    }
    string(varName, options) {
        if (!options.zeroTerminated && !options.length && !options.greedy) {
            throw new Error("One of length, zeroTerminated, or greedy must be defined for string.");
        }
        if ((options.zeroTerminated || options.length) && options.greedy) {
            throw new Error("greedy is mutually exclusive with length and zeroTerminated for string.");
        }
        if (options.stripNull && !(options.length || options.greedy)) {
            throw new Error("length or greedy must be defined if stripNull is enabled.");
        }
        options.encoding = options.encoding || "utf8";
        return this.setNextParser("string", varName, options);
    }
    buffer(varName, options) {
        if (!options.length && !options.readUntil) {
            throw new Error("length or readUntil must be defined for buffer.");
        }
        return this.setNextParser("buffer", varName, options);
    }
    wrapped(varName, options) {
        if (typeof options !== "object" && typeof varName === "object") {
            options = varName;
            varName = "";
        }
        if (!options || !options.wrapper || !options.type) {
            throw new Error("Both wrapper and type must be defined for wrapped.");
        }
        if (!options.length && !options.readUntil) {
            throw new Error("length or readUntil must be defined for wrapped.");
        }
        return this.setNextParser("wrapper", varName, options);
    }
    array(varName, options) {
        if (!options.readUntil && !options.length && !options.lengthInBytes) {
            throw new Error("One of readUntil, length and lengthInBytes must be defined for array.");
        }
        if (!options.type) {
            throw new Error("type is required for array.");
        }
        if (typeof options.type === "string" &&
            !aliasRegistry.has(options.type) &&
            !(options.type in PRIMITIVE_SIZES)) {
            throw new Error(`Array element type "${options.type}" is unkown.`);
        }
        return this.setNextParser("array", varName, options);
    }
    choice(varName, options) {
        if (typeof options !== "object" && typeof varName === "object") {
            options = varName;
            varName = "";
        }
        if (!options) {
            throw new Error("tag and choices are are required for choice.");
        }
        if (!options.tag) {
            throw new Error("tag is requird for choice.");
        }
        if (!options.choices) {
            throw new Error("choices is required for choice.");
        }
        for (const keyString in options.choices) {
            const key = parseInt(keyString, 10);
            const value = options.choices[key];
            if (isNaN(key)) {
                throw new Error(`Choice key "${keyString}" is not a number.`);
            }
            if (typeof value === "string" &&
                !aliasRegistry.has(value) &&
                !(value in PRIMITIVE_SIZES)) {
                throw new Error(`Choice type "${value}" is unkown.`);
            }
        }
        return this.setNextParser("choice", varName, options);
    }
    nest(varName, options) {
        if (typeof options !== "object" && typeof varName === "object") {
            options = varName;
            varName = "";
        }
        if (!options || !options.type) {
            throw new Error("type is required for nest.");
        }
        if (!(options.type instanceof Parser) && !aliasRegistry.has(options.type)) {
            throw new Error("type must be a known parser name or a Parser object.");
        }
        if (!(options.type instanceof Parser) && !varName) {
            throw new Error("type must be a Parser object if the variable name is omitted.");
        }
        return this.setNextParser("nest", varName, options);
    }
    pointer(varName, options) {
        if (!options.offset) {
            throw new Error("offset is required for pointer.");
        }
        if (!options.type) {
            throw new Error("type is required for pointer.");
        }
        if (typeof options.type === "string" &&
            !(options.type in PRIMITIVE_SIZES) &&
            !aliasRegistry.has(options.type)) {
            throw new Error(`Pointer type "${options.type}" is unkown.`);
        }
        return this.setNextParser("pointer", varName, options);
    }
    saveOffset(varName, options = {}) {
        return this.setNextParser("saveOffset", varName, options);
    }
    endianness(endianness) {
        switch (endianness.toLowerCase()) {
            case "little":
                this.endian = "le";
                break;
            case "big":
                this.endian = "be";
                break;
            default:
                throw new Error('endianness must be one of "little" or "big"');
        }
        return this;
    }
    endianess(endianess) {
        return this.endianness(endianess);
    }
    useContextVars(useContextVariables = true) {
        this.useContextVariables = useContextVariables;
        return this;
    }
    create(constructorFn) {
        if (!(constructorFn instanceof Function)) {
            throw new Error("Constructor must be a Function object.");
        }
        this.constructorFn = constructorFn;
        return this;
    }
    getContext(importPath) {
        const ctx = new Context(importPath, this.useContextVariables);
        ctx.pushCode("var dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.length);");
        if (!this.alias) {
            this.addRawCode(ctx);
        }
        else {
            this.addAliasedCode(ctx);
            ctx.pushCode(`return ${FUNCTION_PREFIX + this.alias}(0).result;`);
        }
        return ctx;
    }
    getCode() {
        const importPath = "imports";
        return this.getContext(importPath).code;
    }
    addRawCode(ctx) {
        ctx.pushCode("var offset = 0;");
        ctx.pushCode(`var vars = ${this.constructorFn ? "new constructorFn()" : "{}"};`);
        ctx.pushCode("vars.$parent = null;");
        ctx.pushCode("vars.$root = vars;");
        this.generate(ctx);
        this.resolveReferences(ctx);
        ctx.pushCode("delete vars.$parent;");
        ctx.pushCode("delete vars.$root;");
        ctx.pushCode("return vars;");
    }
    addAliasedCode(ctx) {
        ctx.pushCode(`function ${FUNCTION_PREFIX + this.alias}(offset, context) {`);
        ctx.pushCode(`var vars = ${this.constructorFn ? "new constructorFn()" : "{}"};`);
        ctx.pushCode("var ctx = Object.assign({$parent: null, $root: vars}, context || {});");
        ctx.pushCode(`vars = Object.assign(vars, ctx);`);
        this.generate(ctx);
        ctx.markResolved(this.alias);
        this.resolveReferences(ctx);
        ctx.pushCode("Object.keys(ctx).forEach(function (item) { delete vars[item]; });");
        ctx.pushCode("return { offset: offset, result: vars };");
        ctx.pushCode("}");
        return ctx;
    }
    resolveReferences(ctx) {
        const references = ctx.getUnresolvedReferences();
        ctx.markRequested(references);
        references.forEach((alias) => {
            var _a;
            (_a = aliasRegistry.get(alias)) === null || _a === void 0 ? void 0 : _a.addAliasedCode(ctx);
        });
    }
    compile() {
        const importPath = "imports";
        const ctx = this.getContext(importPath);
        this.compiled = new Function(importPath, "TextDecoder", `return function (buffer, constructorFn) { ${ctx.code} };`)(ctx.imports, TextDecoder);
    }
    sizeOf() {
        let size = NaN;
        if (Object.keys(PRIMITIVE_SIZES).indexOf(this.type) >= 0) {
            size = PRIMITIVE_SIZES[this.type];
            // if this is a fixed length string
        }
        else if (this.type === "string" &&
            typeof this.options.length === "number") {
            size = this.options.length;
            // if this is a fixed length buffer
        }
        else if (this.type === "buffer" &&
            typeof this.options.length === "number") {
            size = this.options.length;
            // if this is a fixed length array
        }
        else if (this.type === "array" &&
            typeof this.options.length === "number") {
            let elementSize = NaN;
            if (typeof this.options.type === "string") {
                elementSize = PRIMITIVE_SIZES[this.options.type];
            }
            else if (this.options.type instanceof Parser) {
                elementSize = this.options.type.sizeOf();
            }
            size = this.options.length * elementSize;
            // if this a skip
        }
        else if (this.type === "seek") {
            size = this.options.length;
            // if this is a nested parser
        }
        else if (this.type === "nest") {
            size = this.options.type.sizeOf();
        }
        else if (!this.type) {
            size = 0;
        }
        if (this.next) {
            size += this.next.sizeOf();
        }
        return size;
    }
    // Follow the parser chain till the root and start parsing from there
    parse(buffer) {
        if (!this.compiled) {
            this.compile();
        }
        return this.compiled(buffer, this.constructorFn);
    }
    setNextParser(type, varName, options) {
        const parser = new Parser();
        parser.type = type;
        parser.varName = varName;
        parser.options = options;
        parser.endian = this.endian;
        if (this.head) {
            this.head.next = parser;
        }
        else {
            this.next = parser;
        }
        this.head = parser;
        return this;
    }
    // Call code generator for this parser
    generate(ctx) {
        if (this.type) {
            switch (this.type) {
                case "uint8":
                case "uint16le":
                case "uint16be":
                case "uint32le":
                case "uint32be":
                case "int8":
                case "int16le":
                case "int16be":
                case "int32le":
                case "int32be":
                case "int64be":
                case "int64le":
                case "uint64be":
                case "uint64le":
                case "floatle":
                case "floatbe":
                case "doublele":
                case "doublebe":
                    this.primitiveGenerateN(this.type, ctx);
                    break;
                case "bit":
                    this.generateBit(ctx);
                    break;
                case "string":
                    this.generateString(ctx);
                    break;
                case "buffer":
                    this.generateBuffer(ctx);
                    break;
                case "seek":
                    this.generateSeek(ctx);
                    break;
                case "nest":
                    this.generateNest(ctx);
                    break;
                case "array":
                    this.generateArray(ctx);
                    break;
                case "choice":
                    this.generateChoice(ctx);
                    break;
                case "pointer":
                    this.generatePointer(ctx);
                    break;
                case "saveOffset":
                    this.generateSaveOffset(ctx);
                    break;
                case "wrapper":
                    this.generateWrapper(ctx);
                    break;
            }
            if (this.type !== "bit")
                this.generateAssert(ctx);
        }
        const varName = ctx.generateVariable(this.varName);
        if (this.options.formatter && this.type !== "bit") {
            this.generateFormatter(ctx, varName, this.options.formatter);
        }
        return this.generateNext(ctx);
    }
    generateAssert(ctx) {
        if (!this.options.assert) {
            return;
        }
        const varName = ctx.generateVariable(this.varName);
        switch (typeof this.options.assert) {
            case "function":
                {
                    const func = ctx.addImport(this.options.assert);
                    ctx.pushCode(`if (!${func}.call(vars, ${varName})) {`);
                }
                break;
            case "number":
                ctx.pushCode(`if (${this.options.assert} !== ${varName}) {`);
                break;
            case "string":
                ctx.pushCode(`if (${JSON.stringify(this.options.assert)} !== ${varName}) {`);
                break;
            default:
                throw new Error("assert option must be a string, number or a function.");
        }
        ctx.generateError(`"Assertion error: ${varName} is " + ${JSON.stringify(this.options.assert.toString())}`);
        ctx.pushCode("}");
    }
    // Recursively call code generators and append results
    generateNext(ctx) {
        if (this.next) {
            ctx = this.next.generate(ctx);
        }
        return ctx;
    }
    generateBit(ctx) {
        // TODO find better method to handle nested bit fields
        const parser = JSON.parse(JSON.stringify(this));
        parser.options = this.options;
        parser.generateAssert = this.generateAssert.bind(this);
        parser.generateFormatter = this.generateFormatter.bind(this);
        parser.varName = ctx.generateVariable(parser.varName);
        ctx.bitFields.push(parser);
        if (!this.next ||
            (this.next && ["bit", "nest"].indexOf(this.next.type) < 0)) {
            const val = ctx.generateTmpVariable();
            ctx.pushCode(`var ${val} = 0;`);
            const getMaxBits = (from = 0) => {
                let sum = 0;
                for (let i = from; i < ctx.bitFields.length; i++) {
                    const length = ctx.bitFields[i].options.length;
                    if (sum + length > 32)
                        break;
                    sum += length;
                }
                return sum;
            };
            const getBytes = (sum) => {
                if (sum <= 8) {
                    ctx.pushCode(`${val} = dataView.getUint8(offset);`);
                    sum = 8;
                }
                else if (sum <= 16) {
                    ctx.pushCode(`${val} = dataView.getUint16(offset);`);
                    sum = 16;
                }
                else if (sum <= 24) {
                    ctx.pushCode(`${val} = (dataView.getUint16(offset) << 8) | dataView.getUint8(offset + 2);`);
                    sum = 24;
                }
                else {
                    ctx.pushCode(`${val} = dataView.getUint32(offset);`);
                    sum = 32;
                }
                ctx.pushCode(`offset += ${sum / 8};`);
                return sum;
            };
            let bitOffset = 0;
            const isBigEndian = this.endian === "be";
            let sum = 0;
            let rem = 0;
            ctx.bitFields.forEach((parser, i) => {
                let length = parser.options.length;
                if (length > rem) {
                    if (rem) {
                        const mask = -1 >>> (32 - rem);
                        ctx.pushCode(`${parser.varName} = (${val} & 0x${mask.toString(16)}) << ${length - rem};`);
                        length -= rem;
                    }
                    bitOffset = 0;
                    rem = sum = getBytes(getMaxBits(i) - rem);
                }
                const offset = isBigEndian ? sum - bitOffset - length : bitOffset;
                const mask = -1 >>> (32 - length);
                ctx.pushCode(`${parser.varName} ${length < parser.options.length ? "|=" : "="} ${val} >> ${offset} & 0x${mask.toString(16)};`);
                // Ensure value is unsigned
                if (parser.options.length === 32) {
                    ctx.pushCode(`${parser.varName} >>>= 0`);
                }
                if (parser.options.assert) {
                    parser.generateAssert(ctx);
                }
                if (parser.options.formatter) {
                    parser.generateFormatter(ctx, parser.varName, parser.options.formatter);
                }
                bitOffset += length;
                rem -= length;
            });
            ctx.bitFields = [];
        }
    }
    generateSeek(ctx) {
        const length = ctx.generateOption(this.options.length);
        ctx.pushCode(`offset += ${length};`);
    }
    generateString(ctx) {
        const name = ctx.generateVariable(this.varName);
        const start = ctx.generateTmpVariable();
        const encoding = this.options.encoding;
        const isHex = encoding.toLowerCase() === "hex";
        const toHex = 'b => b.toString(16).padStart(2, "0")';
        if (this.options.length && this.options.zeroTerminated) {
            const len = this.options.length;
            ctx.pushCode(`var ${start} = offset;`);
            ctx.pushCode(`while(dataView.getUint8(offset++) !== 0 && offset - ${start} < ${len});`);
            const end = `offset - ${start} < ${len} ? offset - 1 : offset`;
            ctx.pushCode(isHex
                ? `${name} = Array.from(buffer.subarray(${start}, ${end}), ${toHex}).join('');`
                : `${name} = new TextDecoder('${encoding}').decode(buffer.subarray(${start}, ${end}));`);
        }
        else if (this.options.length) {
            const len = ctx.generateOption(this.options.length);
            ctx.pushCode(isHex
                ? `${name} = Array.from(buffer.subarray(offset, offset + ${len}), ${toHex}).join('');`
                : `${name} = new TextDecoder('${encoding}').decode(buffer.subarray(offset, offset + ${len}));`);
            ctx.pushCode(`offset += ${len};`);
        }
        else if (this.options.zeroTerminated) {
            ctx.pushCode(`var ${start} = offset;`);
            ctx.pushCode("while(dataView.getUint8(offset++) !== 0);");
            ctx.pushCode(isHex
                ? `${name} = Array.from(buffer.subarray(${start}, offset - 1), ${toHex}).join('');`
                : `${name} = new TextDecoder('${encoding}').decode(buffer.subarray(${start}, offset - 1));`);
        }
        else if (this.options.greedy) {
            ctx.pushCode(`var ${start} = offset;`);
            ctx.pushCode("while(buffer.length > offset++);");
            ctx.pushCode(isHex
                ? `${name} = Array.from(buffer.subarray(${start}, offset), ${toHex}).join('');`
                : `${name} = new TextDecoder('${encoding}').decode(buffer.subarray(${start}, offset));`);
        }
        if (this.options.stripNull) {
            ctx.pushCode(`${name} = ${name}.replace(/\\x00+$/g, '')`);
        }
    }
    generateBuffer(ctx) {
        const varName = ctx.generateVariable(this.varName);
        if (typeof this.options.readUntil === "function") {
            const pred = this.options.readUntil;
            const start = ctx.generateTmpVariable();
            const cur = ctx.generateTmpVariable();
            ctx.pushCode(`var ${start} = offset;`);
            ctx.pushCode(`var ${cur} = 0;`);
            ctx.pushCode(`while (offset < buffer.length) {`);
            ctx.pushCode(`${cur} = dataView.getUint8(offset);`);
            const func = ctx.addImport(pred);
            ctx.pushCode(`if (${func}.call(${ctx.generateVariable()}, ${cur}, buffer.subarray(offset))) break;`);
            ctx.pushCode(`offset += 1;`);
            ctx.pushCode(`}`);
            ctx.pushCode(`${varName} = buffer.subarray(${start}, offset);`);
        }
        else if (this.options.readUntil === "eof") {
            ctx.pushCode(`${varName} = buffer.subarray(offset);`);
        }
        else {
            const len = ctx.generateOption(this.options.length);
            ctx.pushCode(`${varName} = buffer.subarray(offset, offset + ${len});`);
            ctx.pushCode(`offset += ${len};`);
        }
        if (this.options.clone) {
            ctx.pushCode(`${varName} = buffer.constructor.from(${varName});`);
        }
    }
    generateArray(ctx) {
        const length = ctx.generateOption(this.options.length);
        const lengthInBytes = ctx.generateOption(this.options.lengthInBytes);
        const type = this.options.type;
        const counter = ctx.generateTmpVariable();
        const lhs = ctx.generateVariable(this.varName);
        const item = ctx.generateTmpVariable();
        const key = this.options.key;
        const isHash = typeof key === "string";
        if (isHash) {
            ctx.pushCode(`${lhs} = {};`);
        }
        else {
            ctx.pushCode(`${lhs} = [];`);
        }
        if (typeof this.options.readUntil === "function") {
            ctx.pushCode("do {");
        }
        else if (this.options.readUntil === "eof") {
            ctx.pushCode(`for (var ${counter} = 0; offset < buffer.length; ${counter}++) {`);
        }
        else if (lengthInBytes !== undefined) {
            ctx.pushCode(`for (var ${counter} = offset + ${lengthInBytes}; offset < ${counter}; ) {`);
        }
        else {
            ctx.pushCode(`for (var ${counter} = ${length}; ${counter} > 0; ${counter}--) {`);
        }
        if (typeof type === "string") {
            if (!aliasRegistry.get(type)) {
                const typeName = PRIMITIVE_NAMES[type];
                const littleEndian = PRIMITIVE_LITTLE_ENDIANS[type];
                ctx.pushCode(`var ${item} = dataView.get${typeName}(offset, ${littleEndian});`);
                ctx.pushCode(`offset += ${PRIMITIVE_SIZES[type]};`);
            }
            else {
                const tempVar = ctx.generateTmpVariable();
                ctx.pushCode(`var ${tempVar} = ${FUNCTION_PREFIX + type}(offset, {`);
                if (ctx.useContextVariables) {
                    const parentVar = ctx.generateVariable();
                    ctx.pushCode(`$parent: ${parentVar},`);
                    ctx.pushCode(`$root: ${parentVar}.$root,`);
                    if (!this.options.readUntil && lengthInBytes === undefined) {
                        ctx.pushCode(`$index: ${length} - ${counter},`);
                    }
                }
                ctx.pushCode(`});`);
                ctx.pushCode(`var ${item} = ${tempVar}.result; offset = ${tempVar}.offset;`);
                if (type !== this.alias)
                    ctx.addReference(type);
            }
        }
        else if (type instanceof Parser) {
            ctx.pushCode(`var ${item} = {};`);
            const parentVar = ctx.generateVariable();
            ctx.pushScope(item);
            if (ctx.useContextVariables) {
                ctx.pushCode(`${item}.$parent = ${parentVar};`);
                ctx.pushCode(`${item}.$root = ${parentVar}.$root;`);
                if (!this.options.readUntil && lengthInBytes === undefined) {
                    ctx.pushCode(`${item}.$index = ${length} - ${counter};`);
                }
            }
            type.generate(ctx);
            if (ctx.useContextVariables) {
                ctx.pushCode(`delete ${item}.$parent;`);
                ctx.pushCode(`delete ${item}.$root;`);
                ctx.pushCode(`delete ${item}.$index;`);
            }
            ctx.popScope();
        }
        if (isHash) {
            ctx.pushCode(`${lhs}[${item}.${key}] = ${item};`);
        }
        else {
            ctx.pushCode(`${lhs}.push(${item});`);
        }
        ctx.pushCode("}");
        if (typeof this.options.readUntil === "function") {
            const pred = this.options.readUntil;
            const func = ctx.addImport(pred);
            ctx.pushCode(`while (!${func}.call(${ctx.generateVariable()}, ${item}, buffer.subarray(offset)));`);
        }
    }
    generateChoiceCase(ctx, varName, type) {
        if (typeof type === "string") {
            const varName = ctx.generateVariable(this.varName);
            if (!aliasRegistry.has(type)) {
                const typeName = PRIMITIVE_NAMES[type];
                const littleEndian = PRIMITIVE_LITTLE_ENDIANS[type];
                ctx.pushCode(`${varName} = dataView.get${typeName}(offset, ${littleEndian});`);
                ctx.pushCode(`offset += ${PRIMITIVE_SIZES[type]}`);
            }
            else {
                const tempVar = ctx.generateTmpVariable();
                ctx.pushCode(`var ${tempVar} = ${FUNCTION_PREFIX + type}(offset, {`);
                if (ctx.useContextVariables) {
                    ctx.pushCode(`$parent: ${varName}.$parent,`);
                    ctx.pushCode(`$root: ${varName}.$root,`);
                }
                ctx.pushCode(`});`);
                ctx.pushCode(`${varName} = ${tempVar}.result; offset = ${tempVar}.offset;`);
                if (type !== this.alias)
                    ctx.addReference(type);
            }
        }
        else if (type instanceof Parser) {
            ctx.pushPath(varName);
            type.generate(ctx);
            ctx.popPath(varName);
        }
    }
    generateChoice(ctx) {
        const tag = ctx.generateOption(this.options.tag);
        const nestVar = ctx.generateVariable(this.varName);
        if (this.varName) {
            ctx.pushCode(`${nestVar} = {};`);
            if (ctx.useContextVariables) {
                const parentVar = ctx.generateVariable();
                ctx.pushCode(`${nestVar}.$parent = ${parentVar};`);
                ctx.pushCode(`${nestVar}.$root = ${parentVar}.$root;`);
            }
        }
        ctx.pushCode(`switch(${tag}) {`);
        for (const tagString in this.options.choices) {
            const tag = parseInt(tagString, 10);
            const type = this.options.choices[tag];
            ctx.pushCode(`case ${tag}:`);
            this.generateChoiceCase(ctx, this.varName, type);
            ctx.pushCode("break;");
        }
        ctx.pushCode("default:");
        if (this.options.defaultChoice) {
            this.generateChoiceCase(ctx, this.varName, this.options.defaultChoice);
        }
        else {
            ctx.generateError(`"Met undefined tag value " + ${tag} + " at choice"`);
        }
        ctx.pushCode("}");
        if (this.varName && ctx.useContextVariables) {
            ctx.pushCode(`delete ${nestVar}.$parent;`);
            ctx.pushCode(`delete ${nestVar}.$root;`);
        }
    }
    generateNest(ctx) {
        const nestVar = ctx.generateVariable(this.varName);
        if (this.options.type instanceof Parser) {
            if (this.varName) {
                ctx.pushCode(`${nestVar} = {};`);
                if (ctx.useContextVariables) {
                    const parentVar = ctx.generateVariable();
                    ctx.pushCode(`${nestVar}.$parent = ${parentVar};`);
                    ctx.pushCode(`${nestVar}.$root = ${parentVar}.$root;`);
                }
            }
            ctx.pushPath(this.varName);
            this.options.type.generate(ctx);
            ctx.popPath(this.varName);
            if (this.varName && ctx.useContextVariables) {
                if (ctx.useContextVariables) {
                    ctx.pushCode(`delete ${nestVar}.$parent;`);
                    ctx.pushCode(`delete ${nestVar}.$root;`);
                }
            }
        }
        else if (aliasRegistry.has(this.options.type)) {
            const tempVar = ctx.generateTmpVariable();
            ctx.pushCode(`var ${tempVar} = ${FUNCTION_PREFIX + this.options.type}(offset, {`);
            if (ctx.useContextVariables) {
                const parentVar = ctx.generateVariable();
                ctx.pushCode(`$parent: ${parentVar},`);
                ctx.pushCode(`$root: ${parentVar}.$root,`);
            }
            ctx.pushCode(`});`);
            ctx.pushCode(`${nestVar} = ${tempVar}.result; offset = ${tempVar}.offset;`);
            if (this.options.type !== this.alias) {
                ctx.addReference(this.options.type);
            }
        }
    }
    generateWrapper(ctx) {
        const wrapperVar = ctx.generateVariable(this.varName);
        const wrappedBuf = ctx.generateTmpVariable();
        if (typeof this.options.readUntil === "function") {
            const pred = this.options.readUntil;
            const start = ctx.generateTmpVariable();
            const cur = ctx.generateTmpVariable();
            ctx.pushCode(`var ${start} = offset;`);
            ctx.pushCode(`var ${cur} = 0;`);
            ctx.pushCode(`while (offset < buffer.length) {`);
            ctx.pushCode(`${cur} = dataView.getUint8(offset);`);
            const func = ctx.addImport(pred);
            ctx.pushCode(`if (${func}.call(${ctx.generateVariable()}, ${cur}, buffer.subarray(offset))) break;`);
            ctx.pushCode(`offset += 1;`);
            ctx.pushCode(`}`);
            ctx.pushCode(`${wrappedBuf} = buffer.subarray(${start}, offset);`);
        }
        else if (this.options.readUntil === "eof") {
            ctx.pushCode(`${wrappedBuf} = buffer.subarray(offset);`);
        }
        else {
            const len = ctx.generateOption(this.options.length);
            ctx.pushCode(`${wrappedBuf} = buffer.subarray(offset, offset + ${len});`);
            ctx.pushCode(`offset += ${len};`);
        }
        if (this.options.clone) {
            ctx.pushCode(`${wrappedBuf} = buffer.constructor.from(${wrappedBuf});`);
        }
        const tempBuf = ctx.generateTmpVariable();
        const tempOff = ctx.generateTmpVariable();
        const tempView = ctx.generateTmpVariable();
        const func = ctx.addImport(this.options.wrapper);
        ctx.pushCode(`${wrappedBuf} = ${func}.call(this, ${wrappedBuf}).subarray(0);`);
        ctx.pushCode(`var ${tempBuf} = buffer;`);
        ctx.pushCode(`var ${tempOff} = offset;`);
        ctx.pushCode(`var ${tempView} = dataView;`);
        ctx.pushCode(`buffer = ${wrappedBuf};`);
        ctx.pushCode(`offset = 0;`);
        ctx.pushCode(`dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.length);`);
        if (this.options.type instanceof Parser) {
            if (this.varName) {
                ctx.pushCode(`${wrapperVar} = {};`);
            }
            ctx.pushPath(this.varName);
            this.options.type.generate(ctx);
            ctx.popPath(this.varName);
        }
        else if (aliasRegistry.has(this.options.type)) {
            const tempVar = ctx.generateTmpVariable();
            ctx.pushCode(`var ${tempVar} = ${FUNCTION_PREFIX + this.options.type}(0);`);
            ctx.pushCode(`${wrapperVar} = ${tempVar}.result;`);
            if (this.options.type !== this.alias) {
                ctx.addReference(this.options.type);
            }
        }
        ctx.pushCode(`buffer = ${tempBuf};`);
        ctx.pushCode(`dataView = ${tempView};`);
        ctx.pushCode(`offset = ${tempOff};`);
    }
    generateFormatter(ctx, varName, formatter) {
        if (typeof formatter === "function") {
            const func = ctx.addImport(formatter);
            ctx.pushCode(`${varName} = ${func}.call(${ctx.generateVariable()}, ${varName});`);
        }
    }
    generatePointer(ctx) {
        const type = this.options.type;
        const offset = ctx.generateOption(this.options.offset);
        const tempVar = ctx.generateTmpVariable();
        const nestVar = ctx.generateVariable(this.varName);
        // Save current offset
        ctx.pushCode(`var ${tempVar} = offset;`);
        // Move offset
        ctx.pushCode(`offset = ${offset};`);
        if (this.options.type instanceof Parser) {
            ctx.pushCode(`${nestVar} = {};`);
            if (ctx.useContextVariables) {
                const parentVar = ctx.generateVariable();
                ctx.pushCode(`${nestVar}.$parent = ${parentVar};`);
                ctx.pushCode(`${nestVar}.$root = ${parentVar}.$root;`);
            }
            ctx.pushPath(this.varName);
            this.options.type.generate(ctx);
            ctx.popPath(this.varName);
            if (ctx.useContextVariables) {
                ctx.pushCode(`delete ${nestVar}.$parent;`);
                ctx.pushCode(`delete ${nestVar}.$root;`);
            }
        }
        else if (aliasRegistry.has(this.options.type)) {
            const tempVar = ctx.generateTmpVariable();
            ctx.pushCode(`var ${tempVar} = ${FUNCTION_PREFIX + this.options.type}(offset, {`);
            if (ctx.useContextVariables) {
                const parentVar = ctx.generateVariable();
                ctx.pushCode(`$parent: ${parentVar},`);
                ctx.pushCode(`$root: ${parentVar}.$root,`);
            }
            ctx.pushCode(`});`);
            ctx.pushCode(`${nestVar} = ${tempVar}.result; offset = ${tempVar}.offset;`);
            if (this.options.type !== this.alias) {
                ctx.addReference(this.options.type);
            }
        }
        else if (Object.keys(PRIMITIVE_SIZES).indexOf(this.options.type) >= 0) {
            const typeName = PRIMITIVE_NAMES[type];
            const littleEndian = PRIMITIVE_LITTLE_ENDIANS[type];
            ctx.pushCode(`${nestVar} = dataView.get${typeName}(offset, ${littleEndian});`);
            ctx.pushCode(`offset += ${PRIMITIVE_SIZES[type]};`);
        }
        // Restore offset
        ctx.pushCode(`offset = ${tempVar};`);
    }
    generateSaveOffset(ctx) {
        const varName = ctx.generateVariable(this.varName);
        ctx.pushCode(`${varName} = offset`);
    }
}

function newParser() {
    return new Parser().endianess("little");
}
function componentHeaderParser() {
    const limbParser = newParser()
        .uint16("from")
        .uint16("to");
    const colorParser = newParser()
        .uint16("R")
        .uint16("G")
        .uint16("B");
    const strParser = newParser()
        .uint16("_chars")
        .string("text", { length: "_chars" });
    return newParser()
        .uint16("_name")
        .string("name", { length: "_name" })
        .uint16("_format")
        .string("format", { length: "_format" })
        .uint16("_points")
        .uint16("_limbs")
        .uint16("_colors")
        .array("points", {
        type: strParser,
        formatter: (arr) => arr.map((item) => item.text),
        length: "_points"
    })
        .array("limbs", {
        type: limbParser,
        length: "_limbs"
    })
        .array("colors", {
        type: colorParser,
        length: "_colors"
    });
}
function getHeaderParser() {
    const componentParser = componentHeaderParser();
    return newParser()
        .floatle("version")
        .uint16("width")
        .uint16("height")
        .uint16("depth")
        .uint16("_components")
        .array("components", {
        type: componentParser,
        length: "_components"
    })
        // @ts-ignore
        .saveOffset('headerLength');
}
function getBodyParserV0_0(header) {
    let personParser = newParser()
        .int16("id");
    header.components.forEach(component => {
        let pointParser = newParser();
        Array.from(component.format).forEach(c => {
            pointParser = pointParser.floatle(c);
        });
        personParser = personParser.array(component.name, {
            "type": pointParser,
            "length": component._points
        });
    });
    const frameParser = newParser()
        .uint16("_people")
        .array("people", {
        type: personParser,
        length: "_people"
    });
    return newParser()
        .seek(header.headerLength)
        .uint16("fps")
        .uint16("_frames")
        .array("frames", {
        type: frameParser,
        length: "_frames"
    });
}
function parseBodyV0_0(header, buffer) {
    return getBodyParserV0_0(header).parse(buffer);
}
function parseBodyV0_1(header, buffer) {
    const _points = header.components.map(c => c.points.length).reduce((a, b) => a + b, 0);
    const _dims = Math.max(...header.components.map(c => c.format.length)) - 1;
    const infoParser = newParser()
        .seek(header.headerLength)
        .uint16("fps")
        .uint16("_frames")
        .uint16("_people");
    const info = infoParser.parse(buffer);
    // Issue https://github.com/keichi/binary-parser/issues/208
    const parseFloat32Array = (length, offset) => {
        const dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.length);
        let currentOffset = offset;
        const vars = {
            data: new Float32Array(length),
            offset: 0
        };
        for (let i = 0; i < vars.data.length; i++) {
            let $tmp1 = dataView.getFloat32(currentOffset, true);
            currentOffset += 4;
            vars.data[i] = $tmp1;
        }
        vars.offset = currentOffset;
        return vars;
    };
    const data = parseFloat32Array(info._frames * info._people * _points * _dims, header.headerLength + 6);
    const confidence = parseFloat32Array(info._frames * info._people * _points, data.offset);
    function frameRepresentation(i) {
        const people = new Array(info._people);
        for (let j = 0; j < info._people; j++) {
            const person = {};
            people[j] = person;
            let k = 0;
            header.components.forEach(component => {
                person[component.name] = [];
                for (let l = 0; l < component.points.length; l++) {
                    const offset = i * (info._people * _points) + j * _points;
                    const place = offset + k + l;
                    const point = { "C": confidence.data[place] };
                    [...component.format].forEach((dim, dimIndex) => {
                        if (dim !== "C") {
                            point[dim] = data.data[place * _dims + dimIndex];
                        }
                    });
                    person[component.name].push(point);
                }
                k += component.points.length;
            });
        }
        return { people };
    }
    const frames = new Proxy({}, {
        get: function (target, name) {
            if (name === 'length') {
                return info._frames;
            }
            return frameRepresentation(name);
        }
    });
    return Object.assign(Object.assign({}, info), { frames });
}
const headerParser = getHeaderParser();
function parsePose(buffer) {
    const header = headerParser.parse(buffer);
    let body;
    const version = Math.round(header.version * 1000) / 1000;
    switch (version) {
        case 0:
            body = parseBodyV0_0(header, buffer);
            break;
        case 0.1:
            body = parseBodyV0_1(header, buffer);
            break;
        default:
            throw new Error("Parsing this body version is not implemented - " + header.version);
    }
    return { header, body };
}

const empty = {};

const fs = /*#__PURE__*/Object.freeze({
  __proto__: null,
  'default': empty
});

class Pose {
    constructor(header, body) {
        this.header = header;
        this.body = body;
    }
    static from(buffer) {
        const pose = parsePose(buffer);
        return new Pose(pose.header, pose.body);
    }
    static async fromLocal(path) {
        const buffer = undefined(path);
        return Pose.from(buffer);
    }
    static async fromRemote(url, abortController) {
        var _a;
        const init = {};
        if (abortController) {
            init.signal = abortController.signal;
        }
        const res = await fetch(url, init);
        if (!res.ok) {
            let message = (_a = res.statusText) !== null && _a !== void 0 ? _a : String(res.status);
            try {
                const json = await res.json();
                message = json.message;
            }
            catch (e) {
            }
            throw new Error(message);
        }
        const buffer = Buffer.from(await res.arrayBuffer());
        return Pose.from(buffer);
    }
}

class PoseRenderer {
  constructor(viewer) {
    this.viewer = viewer;
  }
  x(v) {
    const n = v * (this.viewer.elWidth - 2 * this.viewer.elPadding.width);
    return n / this.viewer.pose.header.width + this.viewer.elPadding.width;
  }
  y(v) {
    const n = v * (this.viewer.elHeight - 2 * this.viewer.elPadding.height);
    return n / this.viewer.pose.header.height + this.viewer.elPadding.height;
  }
  isJointValid(joint) {
    return joint.C > 0;
  }
  renderJoints(joints, colors) {
    return joints
      .filter(this.isJointValid.bind(this))
      .map((joint, i) => {
      return this.renderJoint(i, joint, colors[i % colors.length]);
    });
  }
  renderLimbs(limbs, joints, colors) {
    return limbs.map(({ from, to }) => {
      const a = joints[from];
      const b = joints[to];
      if (!this.isJointValid(a) || !this.isJointValid(b)) {
        return "";
      }
      const c1 = colors[from % colors.length];
      const c2 = colors[to % colors.length];
      const color = {
        R: (c1.R + c2.R) / 2,
        G: (c1.G + c2.G) / 2,
        B: (c1.B + c2.B) / 2,
      };
      return this.renderLimb(a, b, color);
    });
  }
  renderFrame(frame) {
    return frame.people.map(person => this.viewer.pose.header.components.map(component => {
      const joints = person[component.name];
      return [
        this.renderJoints(joints, component.colors),
        this.renderLimbs(component.limbs, joints, component.colors),
      ];
    }));
  }
}

class SVGPoseRenderer extends PoseRenderer {
  renderJoint(i, joint, color) {
    const { R, G, B } = color;
    return (h("circle", { cx: joint.X, cy: joint.Y, r: 4, class: "joint draggable", style: ({
        fill: `rgb(${R}, ${G}, ${B})`,
        opacity: String(joint.C)
      }), "data-id": i }));
  }
  renderLimb(from, to, color) {
    const { R, G, B } = color;
    return (h("line", { x1: from.X, y1: from.Y, x2: to.X, y2: to.Y, style: {
        stroke: `rgb(${R}, ${G}, ${B})`,
        opacity: String((from.C + to.C) / 2)
      } }));
  }
  render(frame) {
    const viewBox = `0 0 ${this.viewer.pose.header.width} ${this.viewer.pose.header.height}`;
    return (h("svg", { xmlns: "http://www.w3.org/2000/svg", viewBox: viewBox, width: this.viewer.elWidth, height: this.viewer.elHeight },
      h("g", null, this.renderFrame(frame))));
  }
}

class CanvasPoseRenderer extends PoseRenderer {
  renderJoint(_, joint, color) {
    const { R, G, B } = color;
    this.ctx.strokeStyle = `rgba(0, 0, 0, 0)`;
    this.ctx.fillStyle = `rgba(${R}, ${G}, ${B}, ${joint.C})`;
    const radius = Math.round(this.thickness / 3);
    this.ctx.beginPath();
    this.ctx.arc(this.x(joint.X), this.y(joint.Y), radius, 0, 2 * Math.PI);
    this.ctx.fill();
    this.ctx.stroke();
  }
  renderLimb(from, to, color) {
    const { R, G, B } = color;
    this.ctx.lineWidth = this.thickness * 5 / 4;
    this.ctx.strokeStyle = `rgba(${R}, ${G}, ${B}, ${(from.C + to.C) / 2})`;
    this.ctx.beginPath();
    this.ctx.moveTo(this.x(from.X), this.y(from.Y));
    this.ctx.lineTo(this.x(to.X), this.y(to.Y));
    this.ctx.stroke();
  }
  // renderLimb(from: PosePointModel, to: PosePointModel, color: RGBColor) {
  //   const {R, G, B} = color;
  //   this.ctx.fillStyle = `rgba(${R}, ${G}, ${B}, ${(from.C + to.C) / 2})`;
  //
  //   const x = this.x((from.X + to.X) / 2);
  //   const y = this.y((from.Y + to.Y) / 2);
  //
  //   const sub = {x: this.x(from.X - to.X), y: this.y(from.Y - to.Y)}
  //
  //   const length = Math.sqrt(Math.pow(sub.x, 2) + Math.pow(sub.y, 2));
  //   const radiusX = Math.floor(length / 2);
  //   const radiusY = this.thickness;
  //   const rotation = Math.floor(Math.atan2(sub.y, sub.x) * 180 / Math.PI);
  //   this.ctx.beginPath();
  //   this.ctx.ellipse(x, y, radiusX, radiusY, rotation, 0, 360);
  //   this.ctx.fill();
  // }
  render(frame) {
    const drawCanvas = () => {
      var _a;
      const canvas = this.viewer.element.shadowRoot.querySelector('canvas');
      if (canvas) {
        // TODO: this should be unnecessary, but stencil doesn't apply attributes
        canvas.width = this.viewer.elWidth;
        canvas.height = this.viewer.elHeight;
        this.ctx = canvas.getContext('2d');
        if (this.viewer.background) {
          this.ctx.fillStyle = this.viewer.background;
          this.ctx.fillRect(0, 0, canvas.width, canvas.height);
        }
        else {
          this.ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
        const w = this.viewer.elWidth - 2 * this.viewer.elPadding.width;
        const h = this.viewer.elHeight - 2 * this.viewer.elPadding.height;
        this.thickness = (_a = this.viewer.thickness) !== null && _a !== void 0 ? _a : Math.round(Math.sqrt(w * h) / 150);
        this.renderFrame(frame);
      }
      else {
        throw new Error("Canvas isn't available before first render");
      }
    };
    try {
      drawCanvas();
    }
    catch (e) {
      requestAnimationFrame(drawCanvas);
    }
    return (h("canvas", { width: this.viewer.elWidth, height: this.viewer.elHeight }));
  }
}

const poseViewerCss = ":host{display:inline-block}svg,canvas{max-width:100%}svg circle{stroke:black;stroke-width:1px;opacity:0.8}svg line{stroke-width:8px;opacity:0.8;stroke:black}canvas{display:block}";

const PoseViewer = class {
  constructor(hostRef) {
    registerInstance(this, hostRef);
    this.canplaythrough$ = createEvent(this, "canplaythrough$", 7);
    this.ended$ = createEvent(this, "ended$", 7);
    this.loadeddata$ = createEvent(this, "loadeddata$", 7);
    this.loadedmetadata$ = createEvent(this, "loadedmetadata$", 7);
    this.loadstart$ = createEvent(this, "loadstart$", 7);
    this.pause$ = createEvent(this, "pause$", 7);
    this.play$ = createEvent(this, "play$", 7);
    this.firstRender$ = createEvent(this, "firstRender$", 7);
    this.render$ = createEvent(this, "render$", 7);
    // @Event() ratechange$: EventEmitter<void>;
    // @Event() seeked$: EventEmitter<void>;
    // @Event() seeking$: EventEmitter<void>;
    // @Event() timeupdate$: EventEmitter<void>;
    this.hasRendered = false;
    this.src = undefined;
    this.svg = false;
    this.width = null;
    this.height = null;
    this.aspectRatio = null;
    this.padding = null;
    this.thickness = null;
    this.background = null;
    this.loop = false;
    this.autoplay = true;
    this.playbackRate = 1;
    this.currentTime = NaN;
    this.duration = NaN;
    this.ended = false;
    this.paused = true;
    this.readyState = 0;
    this.error = undefined;
  }
  componentWillLoad() {
    this.renderer = this.svg ? new SVGPoseRenderer(this) : new CanvasPoseRenderer(this);
    return this.srcChange();
  }
  componentDidLoad() {
    this.resizeObserver = new ResizeObserver(this.setDimensions.bind(this));
    this.resizeObserver.observe(this.element);
  }
  async getRemotePose() {
    // Abort previous request
    if (this.fetchAbortController) {
      this.fetchAbortController.abort();
    }
    this.fetchAbortController = new AbortController();
    this.pose = await Pose.fromRemote(this.src, this.fetchAbortController);
  }
  initPose() {
    this.setDimensions();
    // Loaded done events
    this.loadedmetadata$.emit();
    this.loadeddata$.emit();
    this.canplaythrough$.emit();
    this.duration = (this.pose.body.frames.length - 1) / this.pose.body.fps;
    this.currentTime = 0;
    if (this.autoplay) {
      this.play();
    }
  }
  async srcChange() {
    // Can occur from both an attribute change AND componentWillLoad event
    if (this.src === this.lastSrc) {
      return;
    }
    this.lastSrc = this.src;
    // Clear previous pose
    this.clearInterval();
    this.setDimensions();
    delete this.pose;
    this.currentTime = NaN;
    this.duration = NaN;
    this.hasRendered = false;
    if (!this.src) {
      return;
    }
    // Load new pose
    this.ended = false;
    this.loadstart$.emit();
    this.error = null;
    try {
      await this.getRemotePose();
      this.initPose();
      this.error = null;
    }
    catch (e) {
      console.error('PoseViewer error', e);
      this.error = e;
    }
  }
  setDimensions() {
    this.elPadding = { width: 0, height: 0 };
    if (!this.pose) {
      this.elWidth = 0;
      this.elHeight = 0;
      return;
    }
    // When nothing is marked, use pose dimensions
    if (!this.width && !this.height) {
      this.elWidth = this.pose.header.width;
      this.elHeight = this.pose.header.height;
      return;
    }
    const rect = this.element.getBoundingClientRect();
    const parseSize = (size, by) => size.endsWith("px") ? Number(size.slice(0, -2)) : (size.endsWith("%") ? by * size.slice(0, -1) / 100 : Number(size));
    // When both are marked,
    if (this.width && this.height) {
      this.elWidth = parseSize(this.width, rect.width);
      this.elHeight = parseSize(this.height, rect.height);
    }
    else if (this.width) {
      this.elWidth = parseSize(this.width, rect.width);
      this.elHeight = this.aspectRatio ? this.elWidth * this.aspectRatio :
        (this.pose.header.height / this.pose.header.width) * this.elWidth;
    }
    else if (this.height) {
      this.elHeight = parseSize(this.height, rect.height);
      this.elWidth = this.aspectRatio ? this.elHeight / this.aspectRatio :
        (this.pose.header.width / this.pose.header.height) * this.elHeight;
    }
    // General padding
    if (this.padding) {
      this.elPadding.width += parseSize(this.padding, this.elWidth);
      this.elPadding.height += parseSize(this.padding, this.elHeight);
    }
    // Aspect ratio padding
    const ratioWidth = this.elWidth - this.elPadding.width * 2;
    const ratioHeight = this.elHeight - this.elPadding.height * 2;
    const elAR = ratioWidth / ratioHeight;
    const poseAR = this.pose.header.width / this.pose.header.height;
    if (poseAR > elAR) {
      this.elPadding.height += (poseAR - elAR) * ratioHeight / 2;
    }
    else {
      this.elPadding.width += (1 / poseAR - 1 / elAR) * ratioWidth / 2;
    }
  }
  async syncMedia(media) {
    this.media = media;
    this.media.addEventListener('pause', this.pause.bind(this));
    this.media.addEventListener('play', this.play.bind(this));
    const syncTime = () => this.currentTime = this.frameTime(this.media.currentTime);
    this.media.addEventListener('seek', syncTime);
    this.media.addEventListener('timeupdate', syncTime); // To always keep synced
    // Others
    const updateRate = () => this.playbackRate = this.media.playbackRate;
    this.media.addEventListener('ratechange', updateRate);
    updateRate();
    // Start the pose according to the video
    this.clearInterval();
    if (this.media.paused) {
      this.pause();
    }
    else {
      this.play();
    }
  }
  async getPose() {
    return this.pose;
  }
  async nextFrame() {
    const newTime = this.currentTime + 1 / this.pose.body.fps;
    if (newTime > this.duration) {
      if (this.loop) {
        this.currentTime = newTime % this.duration;
      }
      else {
        this.ended$.emit();
        this.ended = true;
      }
    }
    else {
      this.currentTime = newTime;
    }
  }
  frameTime(time) {
    if (!this.pose) {
      return 0;
    }
    return Math.floor(time * this.pose.body.fps) / this.pose.body.fps;
  }
  async play() {
    if (!this.paused) {
      this.clearInterval();
    }
    this.paused = false;
    this.play$.emit();
    // Reset clip if exceeded duration
    if (this.currentTime > this.duration) {
      this.currentTime = 0;
    }
    const intervalTime = 1000 / (this.pose.body.fps * this.playbackRate);
    if (this.media) {
      this.loopInterval = setInterval(() => this.currentTime = this.frameTime(this.media.currentTime), intervalTime);
    }
    else {
      // Add the time passed in an interval.
      let lastTime = Date.now() / 1000;
      this.loopInterval = setInterval(() => {
        const now = Date.now() / 1000;
        this.currentTime += (now - lastTime) * this.playbackRate;
        lastTime = now;
        if (this.currentTime > this.duration) {
          if (this.loop) {
            this.currentTime = this.currentTime % this.duration;
          }
          else {
            this.ended$.emit();
            this.ended = true;
            this.clearInterval();
          }
        }
      }, intervalTime);
    }
  }
  async pause() {
    this.paused = true;
    this.pause$.emit();
    this.clearInterval();
  }
  clearInterval() {
    if (this.loopInterval) {
      clearInterval(this.loopInterval);
    }
  }
  disconnectedCallback() {
    this.clearInterval();
  }
  render() {
    if (this.error) {
      return this.error.name !== "AbortError" ? this.error.message : "";
    }
    if (!this.pose || isNaN(this.currentTime) || !this.renderer) {
      return "";
    }
    const currentTime = this.currentTime > this.duration ? this.duration : this.currentTime;
    const frameId = Math.floor(currentTime * this.pose.body.fps);
    const frame = this.pose.body.frames[frameId];
    const render = this.renderer.render(frame);
    if (!this.hasRendered) {
      requestAnimationFrame(() => {
        this.hasRendered = true;
        this.firstRender$.emit();
      });
    }
    requestAnimationFrame(() => this.render$.emit());
    return (h(Host, null, render));
  }
  get element() { return getElement(this); }
  static get watchers() { return {
    "src": ["srcChange"]
  }; }
};
PoseViewer.style = poseViewerCss;

export { PoseViewer as pose_viewer };

//# sourceMappingURL=pose-viewer.entry.js.map