// The following ~400 lines of code exists for exactly one purpose, which is
// to optimize this code:
//
//     byte_slice.iter().position(|&b| b > 0x7F).unwrap_or(byte_slice.len())
//
// Yes... Overengineered is a word that comes to mind, but this is effectively
// a very similar problem to memchr, and virtually nobody has been able to
// resist optimizing the crap out of that (except for perhaps the BSD and MUSL
// folks). In particular, this routine makes a very common case (ASCII) very
// fast, which seems worth it. AVX2 is selected only when enabled for the
// compilation target, without runtime target feature detection.
//
// In order to understand the SIMD version below, it would be good to read this
// comment describing how my memchr routine works:
// https://github.com/BurntSushi/rust-memchr/blob/b0a29f267f4a7fad8ffcc8fe8377a06498202883/src/x86/sse2.rs#L19-L106
//
// The primary difference with memchr is that for ASCII, we can do a bit less
// work. In particular, we don't need to detect the presence of a specific
// byte, but rather, whether any byte has its most significant bit set. That
// means we can effectively skip the _mm_cmpeq_epi8 step and jump straight to
// _mm_movemask_epi8.

#[cfg(any(test, miri, not(target_arch = "x86_64")))]
const USIZE_BYTES: usize = core::mem::size_of::<usize>();
#[cfg(any(test, miri, not(target_arch = "x86_64")))]
const ALIGN_MASK: usize = core::mem::align_of::<usize>() - 1;
#[cfg(any(test, miri, not(target_arch = "x86_64")))]
const FALLBACK_LOOP_SIZE: usize = 2 * USIZE_BYTES;

// This is a mask where the most significant bit of each byte in the usize
// is set. We test this bit to determine whether a character is ASCII or not.
// Namely, a single byte is regarded as an ASCII codepoint if and only if it's
// most significant bit is not set.
#[cfg(any(test, miri, not(target_arch = "x86_64")))]
const ASCII_MASK_U64: u64 = 0x8080808080808080;
#[cfg(any(test, miri, not(target_arch = "x86_64")))]
const ASCII_MASK: usize = ASCII_MASK_U64 as usize;

/// Returns the index of the first non ASCII byte in the given slice.
///
/// If slice only contains ASCII bytes, then the length of the slice is
/// returned.
pub fn first_non_ascii_byte(slice: &[u8]) -> usize {
    #[cfg(any(miri, not(target_arch = "x86_64")))]
    {
        first_non_ascii_byte_fallback(slice)
    }

    #[cfg(all(not(miri), target_arch = "x86_64"))]
    {
        first_non_ascii_byte_sse2(slice)
    }
}

#[cfg(any(test, miri, not(target_arch = "x86_64")))]
fn first_non_ascii_byte_fallback(slice: &[u8]) -> usize {
    let start_ptr = slice.as_ptr();
    let end_ptr = slice[slice.len()..].as_ptr();
    let mut ptr = start_ptr;

    unsafe {
        if slice.len() < USIZE_BYTES {
            return first_non_ascii_byte_slow(start_ptr, end_ptr, ptr);
        }

        let chunk = read_unaligned_usize(ptr);
        let mask = chunk & ASCII_MASK;
        if mask != 0 {
            return first_non_ascii_byte_mask(mask);
        }

        ptr = ptr_add(ptr, USIZE_BYTES - (start_ptr as usize & ALIGN_MASK));
        debug_assert!(ptr > start_ptr);
        debug_assert!(ptr_sub(end_ptr, USIZE_BYTES) >= start_ptr);
        if slice.len() >= FALLBACK_LOOP_SIZE {
            while ptr <= ptr_sub(end_ptr, FALLBACK_LOOP_SIZE) {
                debug_assert_eq!(0, (ptr as usize) % USIZE_BYTES);

                let a = *(ptr as *const usize);
                let b = *(ptr_add(ptr, USIZE_BYTES) as *const usize);
                if (a | b) & ASCII_MASK != 0 {
                    // What a kludge. We wrap the position finding code into
                    // a non-inlineable function, which makes the codegen in
                    // the tight loop above a bit better by avoiding a
                    // couple extra movs. We pay for it by two additional
                    // stores, but only in the case of finding a non-ASCII
                    // byte.
                    #[inline(never)]
                    unsafe fn findpos(
                        start_ptr: *const u8,
                        ptr: *const u8,
                    ) -> usize {
                        let a = *(ptr as *const usize);
                        let b = *(ptr_add(ptr, USIZE_BYTES) as *const usize);

                        let mut at = sub(ptr, start_ptr);
                        let maska = a & ASCII_MASK;
                        if maska != 0 {
                            return at + first_non_ascii_byte_mask(maska);
                        }

                        at += USIZE_BYTES;
                        let maskb = b & ASCII_MASK;
                        debug_assert!(maskb != 0);
                        return at + first_non_ascii_byte_mask(maskb);
                    }
                    return findpos(start_ptr, ptr);
                }
                ptr = ptr_add(ptr, FALLBACK_LOOP_SIZE);
            }
        }
        first_non_ascii_byte_slow(start_ptr, end_ptr, ptr)
    }
}

#[cfg(all(not(miri), target_arch = "x86_64"))]
fn first_non_ascii_byte_sse2(slice: &[u8]) -> usize {
    use core::arch::x86_64::*;

    const VECTOR_SIZE: usize = core::mem::size_of::<__m128i>();
    const VECTOR_ALIGN: usize = VECTOR_SIZE - 1;
    const VECTOR_LOOP_SIZE: usize = 4 * VECTOR_SIZE;

    let start_ptr = slice.as_ptr();
    let end_ptr = slice[slice.len()..].as_ptr();
    let mut ptr = start_ptr;

    unsafe {
        if slice.len() < VECTOR_SIZE {
            return first_non_ascii_byte_slow(start_ptr, end_ptr, ptr);
        }

        let chunk = _mm_loadu_si128(ptr.cast());
        let mask = _mm_movemask_epi8(chunk);
        if mask != 0 {
            return mask.trailing_zeros() as usize;
        }

        ptr = ptr.add(VECTOR_SIZE - (start_ptr as usize & VECTOR_ALIGN));
        debug_assert!(ptr > start_ptr);
        debug_assert!(end_ptr.sub(VECTOR_SIZE) >= start_ptr);
        if slice.len() >= VECTOR_LOOP_SIZE {
            // Keep short scans on the established SSE2 path. Only targets
            // compiled with AVX2 can delegate longer inputs to the wide loop.
            #[cfg(target_feature = "avx2")]
            if slice.len() >= 1024 {
                // SAFETY: The target enables AVX2 and the length check
                // satisfies the wide loop's 64-byte precondition.
                return first_non_ascii_byte_avx2(slice);
            }
            while ptr <= ptr_sub(end_ptr, VECTOR_LOOP_SIZE) {
                debug_assert_eq!(0, (ptr as usize) % VECTOR_SIZE);

                let a = _mm_load_si128(ptr.cast());
                let b = _mm_load_si128(ptr.add(VECTOR_SIZE).cast());
                let c = _mm_load_si128(ptr.add(2 * VECTOR_SIZE).cast());
                let d = _mm_load_si128(ptr.add(3 * VECTOR_SIZE).cast());

                let or1 = _mm_or_si128(a, b);
                let or2 = _mm_or_si128(c, d);
                let or3 = _mm_or_si128(or1, or2);
                if _mm_movemask_epi8(or3) != 0 {
                    let mut at = sub(ptr, start_ptr);
                    let mask = _mm_movemask_epi8(a);
                    if mask != 0 {
                        return at + mask.trailing_zeros() as usize;
                    }

                    at += VECTOR_SIZE;
                    let mask = _mm_movemask_epi8(b);
                    if mask != 0 {
                        return at + mask.trailing_zeros() as usize;
                    }

                    at += VECTOR_SIZE;
                    let mask = _mm_movemask_epi8(c);
                    if mask != 0 {
                        return at + mask.trailing_zeros() as usize;
                    }

                    at += VECTOR_SIZE;
                    let mask = _mm_movemask_epi8(d);
                    debug_assert!(mask != 0);
                    return at + mask.trailing_zeros() as usize;
                }
                ptr = ptr_add(ptr, VECTOR_LOOP_SIZE);
            }
        }
        while ptr <= end_ptr.sub(VECTOR_SIZE) {
            debug_assert!(sub(end_ptr, ptr) >= VECTOR_SIZE);

            let chunk = _mm_loadu_si128(ptr.cast());
            let mask = _mm_movemask_epi8(chunk);
            if mask != 0 {
                return sub(ptr, start_ptr) + mask.trailing_zeros() as usize;
            }
            ptr = ptr.add(VECTOR_SIZE);
        }
        // SAFETY: The initial length check guarantees at least VECTOR_SIZE
        // bytes. A final vector ending at end_ptr stays within the slice.
        // Any overlap contains bytes already checked as ASCII, so the first
        // set bit still identifies the first non-ASCII byte.
        if ptr < end_ptr {
            let last = end_ptr.sub(VECTOR_SIZE);
            let mask = _mm_movemask_epi8(_mm_loadu_si128(last.cast()));
            if mask != 0 {
                return sub(last, start_ptr) + mask.trailing_zeros() as usize;
            }
        }
        slice.len()
    }
}

// The AVX2 kernel uses 32-byte vectors and 128-byte blocks.
// SAFETY: Its caller must guarantee AVX2 and at least 64 input bytes.
// Two initial unaligned loads advance to a 32-byte-aligned main loop.
// Later loop bounds and the overlapping tail keep every load in the slice.
#[cfg(all(
    not(miri),
    target_arch = "x86_64",
    any(test, target_feature = "avx2"),
))]
#[target_feature(enable = "avx2")]
unsafe fn first_non_ascii_byte_avx2(slice: &[u8]) -> usize {
    use core::arch::x86_64::*;

    const VECTOR_SIZE: usize = core::mem::size_of::<__m256i>();
    const VECTOR_ALIGN: usize = VECTOR_SIZE - 1;
    const VECTOR_LOOP_SIZE: usize = 4 * VECTOR_SIZE;

    let start_ptr = slice.as_ptr();
    let end_ptr = slice[slice.len()..].as_ptr();
    let mut ptr = start_ptr;

    unsafe {
        debug_assert!(slice.len() >= 2 * VECTOR_SIZE);

        let chunk = _mm256_loadu_si256(ptr.cast());
        let mask = _mm256_movemask_epi8(chunk);
        if mask != 0 {
            return mask.trailing_zeros() as usize;
        }

        // Check the next vector before the bulk loop. This avoids scanning
        // a whole block for the short ASCII runs common in mixed UTF-8.
        let chunk = _mm256_loadu_si256(ptr.add(VECTOR_SIZE).cast());
        let mask = _mm256_movemask_epi8(chunk);
        if mask != 0 {
            return VECTOR_SIZE + mask.trailing_zeros() as usize;
        }

        ptr = ptr.add(2 * VECTOR_SIZE - (start_ptr as usize & VECTOR_ALIGN));
        debug_assert!(ptr > start_ptr);
        debug_assert!(end_ptr.sub(VECTOR_SIZE) >= start_ptr);
        if slice.len() >= VECTOR_LOOP_SIZE {
            while ptr <= ptr_sub(end_ptr, VECTOR_LOOP_SIZE) {
                debug_assert_eq!(0, (ptr as usize) % VECTOR_SIZE);

                let a = _mm256_load_si256(ptr.cast());
                let b = _mm256_load_si256(ptr.add(VECTOR_SIZE).cast());
                let c = _mm256_load_si256(ptr.add(2 * VECTOR_SIZE).cast());
                let d = _mm256_load_si256(ptr.add(3 * VECTOR_SIZE).cast());

                let or1 = _mm256_or_si256(a, b);
                let or2 = _mm256_or_si256(c, d);
                let or3 = _mm256_or_si256(or1, or2);
                if _mm256_movemask_epi8(or3) != 0 {
                    let mut at = sub(ptr, start_ptr);
                    let mask = _mm256_movemask_epi8(a);
                    if mask != 0 {
                        return at + mask.trailing_zeros() as usize;
                    }

                    at += VECTOR_SIZE;
                    let mask = _mm256_movemask_epi8(b);
                    if mask != 0 {
                        return at + mask.trailing_zeros() as usize;
                    }

                    at += VECTOR_SIZE;
                    let mask = _mm256_movemask_epi8(c);
                    if mask != 0 {
                        return at + mask.trailing_zeros() as usize;
                    }

                    at += VECTOR_SIZE;
                    let mask = _mm256_movemask_epi8(d);
                    debug_assert!(mask != 0);
                    return at + mask.trailing_zeros() as usize;
                }
                ptr = ptr_add(ptr, VECTOR_LOOP_SIZE);
            }
        }
        while ptr <= end_ptr.sub(VECTOR_SIZE) {
            debug_assert!(sub(end_ptr, ptr) >= VECTOR_SIZE);

            let chunk = _mm256_loadu_si256(ptr.cast());
            let mask = _mm256_movemask_epi8(chunk);
            if mask != 0 {
                return sub(ptr, start_ptr) + mask.trailing_zeros() as usize;
            }
            ptr = ptr.add(VECTOR_SIZE);
        }
        // The initial length check guarantees a full vector. This load
        // ends at end_ptr and overlaps only bytes already checked as ASCII.
        if ptr < end_ptr {
            let last = end_ptr.sub(VECTOR_SIZE);
            let mask = _mm256_movemask_epi8(_mm256_loadu_si256(last.cast()));
            if mask != 0 {
                return sub(last, start_ptr) + mask.trailing_zeros() as usize;
            }
        }
        slice.len()
    }
}

#[inline(always)]
unsafe fn first_non_ascii_byte_slow(
    start_ptr: *const u8,
    end_ptr: *const u8,
    mut ptr: *const u8,
) -> usize {
    debug_assert!(start_ptr <= ptr);
    debug_assert!(ptr <= end_ptr);

    while ptr < end_ptr {
        if *ptr > 0x7F {
            return sub(ptr, start_ptr);
        }
        ptr = ptr.offset(1);
    }
    sub(end_ptr, start_ptr)
}

/// Compute the position of the first ASCII byte in the given mask.
///
/// The mask should be computed by `chunk & ASCII_MASK`, where `chunk` is
/// 8 contiguous bytes of the slice being checked where *at least* one of those
/// bytes is not an ASCII byte.
///
/// The position returned is always in the inclusive range [0, 7].
#[cfg(any(test, miri, not(target_arch = "x86_64")))]
fn first_non_ascii_byte_mask(mask: usize) -> usize {
    #[cfg(target_endian = "little")]
    {
        mask.trailing_zeros() as usize / 8
    }
    #[cfg(target_endian = "big")]
    {
        mask.leading_zeros() as usize / 8
    }
}

/// Increment the given pointer by the given amount.
unsafe fn ptr_add(ptr: *const u8, amt: usize) -> *const u8 {
    ptr.add(amt)
}

/// Decrement the given pointer by the given amount.
unsafe fn ptr_sub(ptr: *const u8, amt: usize) -> *const u8 {
    ptr.sub(amt)
}

#[cfg(any(test, miri, not(target_arch = "x86_64")))]
unsafe fn read_unaligned_usize(ptr: *const u8) -> usize {
    use core::ptr;

    let mut n: usize = 0;
    ptr::copy_nonoverlapping(ptr, &mut n as *mut _ as *mut u8, USIZE_BYTES);
    n
}

/// Subtract `b` from `a` and return the difference. `a` should be greater than
/// or equal to `b`.
fn sub(a: *const u8, b: *const u8) -> usize {
    debug_assert!(a >= b);
    (a as usize) - (b as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(all(not(miri), target_arch = "x86_64"))]
    fn check_x86_kernel(scan: fn(&[u8]) -> usize) {
        for align in 0..64 {
            for len in 0..=257 {
                // Include NUL as ordinary ASCII input.
                let mut bytes = std::vec![0; align + len];
                assert_eq!(scan(&bytes[align..]), len);
                for at in 0..len {
                    for byte in [0x80, 0xFF] {
                        bytes[align + at] = byte;
                        assert_eq!(scan(&bytes[align..]), at,
                            "align: {align}, len: {len}, at: {at}, byte: {byte}");
                    }
                    bytes[align + at] = 0;
                }
            }
        }
        for byte in 0u8..=255 {
            let bytes = [byte; 513];
            let expected = if byte <= 0x7F { bytes.len() } else { 0 };
            assert_eq!(scan(&bytes), expected);
        }
        for first in 0..257 {
            let mut bytes = std::vec![0x7F; 513];
            bytes[first] = 0x80;
            bytes[first + 16] = 0xFF;
            bytes[first + 128] = 0x81;
            assert_eq!(scan(&bytes), first);
        }
    }

    #[test]
    #[cfg(all(not(miri), target_arch = "x86_64"))]
    fn sse2_matches_scalar() {
        check_x86_kernel(first_non_ascii_byte_sse2);
    }

    // Our testing approach here is to try and exhaustively test every case.
    // This includes the position at which a non-ASCII byte occurs in addition
    // to the alignment of the slice that we're searching.

    #[test]
    fn positive_fallback_forward() {
        for i in 0..517 {
            let s = "a".repeat(i);
            assert_eq!(
                i,
                first_non_ascii_byte_fallback(s.as_bytes()),
                "i: {:?}, len: {:?}, s: {:?}",
                i,
                s.len(),
                s
            );
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[cfg(not(miri))]
    fn positive_sse2_forward() {
        for i in 0..517 {
            let b = "a".repeat(i).into_bytes();
            assert_eq!(b.len(), first_non_ascii_byte_sse2(&b));
        }
    }

    #[test]
    #[cfg(not(miri))]
    fn negative_fallback_forward() {
        for i in 0..517 {
            for align in 0..65 {
                let mut s = "a".repeat(i);
                s.push_str("☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃");
                let s = s.get(align..).unwrap_or("");
                assert_eq!(
                    i.saturating_sub(align),
                    first_non_ascii_byte_fallback(s.as_bytes()),
                    "i: {:?}, align: {:?}, len: {:?}, s: {:?}",
                    i,
                    align,
                    s.len(),
                    s
                );
            }
        }
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    #[cfg(not(miri))]
    fn negative_sse2_forward() {
        for i in 0..517 {
            for align in 0..65 {
                let mut s = "a".repeat(i);
                s.push_str("☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃☃");
                let s = s.get(align..).unwrap_or("");
                assert_eq!(
                    i.saturating_sub(align),
                    first_non_ascii_byte_sse2(s.as_bytes()),
                    "i: {:?}, align: {:?}, len: {:?}, s: {:?}",
                    i,
                    align,
                    s.len(),
                    s
                );
            }
        }
    }
    #[cfg(all(not(miri), target_arch = "x86_64"))]
    mod avx2 {
        use super::super::*;

        fn check_x86_kernel(scan: unsafe fn(&[u8]) -> usize, min_len: usize) {
            for align in 0..64 {
                for len in min_len..=257 {
                    // Include NUL as ordinary ASCII input.
                    let mut bytes = std::vec![0; align + len];
                    assert_eq!(unsafe { scan(&bytes[align..]) }, len);
                    for at in 0..len {
                        for byte in [0x80, 0xFF] {
                            bytes[align + at] = byte;
                            assert_eq!(unsafe { scan(&bytes[align..]) }, at,
                                "align: {align}, len: {len}, at: {at}, byte: {byte}");
                        }
                        bytes[align + at] = 0;
                    }
                }
            }
            for byte in 0u8..=255 {
                let bytes = [byte; 513];
                let expected = if byte <= 0x7F { bytes.len() } else { 0 };
                assert_eq!(unsafe { scan(&bytes) }, expected);
            }
            for first in 0..257 {
                let mut bytes = std::vec![0x7F; 513];
                bytes[first] = 0x80;
                bytes[first + 16] = 0xFF;
                bytes[first + 128] = 0x81;
                assert_eq!(unsafe { scan(&bytes) }, first);
            }
        }

        #[test]
        fn avx2_matches_scalar() {
            if !std::is_x86_feature_detected!("avx2") {
                return;
            }
            check_x86_kernel(first_non_ascii_byte_avx2, 64);
        }

        #[test]
        fn x86_dispatch_matches_scalar() {
            check_x86_kernel(first_non_ascii_byte, 0);
        }

        #[test]
        fn large_dispatch_matches_scalar() {
            for len in [1023, 1024, 1025, 1151, 1152, 1153, 2048] {
                for align in 0..64 {
                    let mut bytes = std::vec![0x7F; align + len];
                    assert_eq!(first_non_ascii_byte(&bytes[align..]), len);
                    for at in 0..len {
                        bytes[align + at] = 0x80;
                        assert_eq!(
                            first_non_ascii_byte(&bytes[align..]),
                            at,
                            "align: {align}, len: {len}, at: {at}"
                        );
                        bytes[align + at] = 0x7F;
                    }
                }
            }
        }
    }
}
