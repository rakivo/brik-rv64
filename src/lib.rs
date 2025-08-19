//! RISC-V 64-Bit Extension for brik_rv32: RISC-V 32-bit encoding/decoding crate

#![no_std]
#![warn(
    anonymous_parameters,
    missing_copy_implementations,
    missing_debug_implementations,
    nonstandard_style,
    rust_2018_idioms,
    single_use_lifetimes,
    trivial_casts,
    trivial_numeric_casts,
    unreachable_pub,
    unused_extern_crates,
    unused_qualifications,
    variant_size_differences
)]

use brik_rv32::Reg::{self, *};
use brik_rv32::{I32, AqRl, ConversionError};

use I64::*;

/// 64-Bit assembly instruction
#[allow(clippy::enum_variant_names)]
#[allow(non_camel_case_types)]
#[allow(missing_docs)]
#[derive(Copy, Clone, Debug)]
pub enum I64 {
    //// RV64I Base Instruction Set (64-bit specific loads/stores) ////
    /// I: Load Doubleword (`R[d]: M[R[s] + im]`) - RV64 only
    LD { d: Reg, s: Reg, im: i16 },
    /// S: Store Doubleword - RV64 only
    SD { s1: Reg, s2: Reg, im: i16 },
    /// I: Load Word Unsigned (`R[d]: M[R[s] + im]`) - RV64 only, zero-extends to 64 bits
    LWU { d: Reg, s: Reg, im: i16 },

    //// RV64I-specific instructions (32-bit operations with sign extension) ////
    /// R: Add Word (`R[d]: sext(R[s1][31:0] + R[s2][31:0])`) - RV64 only, 32-bit add with sign extension
    ADDW { d: Reg, s1: Reg, s2: Reg },
    /// R: Subtract Word (`R[d]: sext(R[s1][31:0] - R[s2][31:0])`) - RV64 only, 32-bit subtract with sign extension
    SUBW { d: Reg, s1: Reg, s2: Reg },
    /// I: Add Immediate Word (`R[d]: sext(R[s][31:0] + im)`) - RV64 only, 32-bit add immediate with sign extension
    ADDIW { d: Reg, s: Reg, im: i16 },
    /// R: Shift Left Logical Word (`R[d]: sext(R[s1][31:0] << R[s2][4:0])`) - RV64 only, 32-bit left shift with sign extension
    SLLW { d: Reg, s1: Reg, s2: Reg },
    /// R: Shift Right Logical Word (`R[d]: sext(R[s1][31:0] >> R[s2][4:0])`) - RV64 only, 32-bit logical right shift with sign extension
    SRLW { d: Reg, s1: Reg, s2: Reg },
    /// R: Shift Right Arithmetic Word (`R[d]: sext(R[s1][31:0] >>> R[s2][4:0])`) - RV64 only, 32-bit arithmetic right shift with sign extension
    SRAW { d: Reg, s1: Reg, s2: Reg },
    /// I: Shift Left Logical Immediate Word (`R[d]: sext(R[s][31:0] << shamt)`) - RV64 only, 32-bit left shift immediate with sign extension
    SLLIW { d: Reg, s: Reg, shamt: u8 },
    /// I: Shift Right Logical Immediate Word (`R[d]: sext(R[s][31:0] >> shamt)`) - RV64 only, 32-bit logical right shift immediate with sign extension
    SRLIW { d: Reg, s: Reg, shamt: u8 },
    /// I: Shift Right Arithmetic Immediate Word (`R[d]: sext(R[s][31:0] >>> shamt)`) - RV64 only, 32-bit arithmetic right shift immediate with sign extension
    SRAIW { d: Reg, s: Reg, shamt: u8 },

    //// RV64 M-extension instructions (64-bit variants and 32-bit word operations) ////
    /// R: Multiply Word (`R[d]: sext(R[s1][31:0] * R[s2][31:0])`) - RV64 only, 32-bit multiply with sign extension
    MULW { d: Reg, s1: Reg, s2: Reg },
    /// R: Divide Word (`R[d]: sext(R[s1][31:0] / R[s2][31:0])`) - RV64 only, 32-bit signed division with sign extension
    DIVW { d: Reg, s1: Reg, s2: Reg },
    /// R: Divide Unsigned Word (`R[d]: sext(R[s1][31:0] /u R[s2][31:0])`) - RV64 only, 32-bit unsigned division with sign extension
    DIVUW { d: Reg, s1: Reg, s2: Reg },
    /// R: Remainder Word (`R[d]: sext(R[s1][31:0] % R[s2][31:0])`) - RV64 only, 32-bit signed remainder with sign extension
    REMW { d: Reg, s1: Reg, s2: Reg },
    /// R: Remainder Unsigned Word (`R[d]: sext(R[s1][31:0] %u R[s2][31:0])`) - RV64 only, 32-bit unsigned remainder with sign extension
    REMUW { d: Reg, s1: Reg, s2: Reg },
    /// R: Multiply High Word (`R[d]: sext((R[s1][31:0] * R[s2][31:0])[63:32])`) - RV64 only, upper 32 bits of 32-bit signed multiply, sign-extended
    MULHW { d: Reg, s1: Reg, s2: Reg },
    /// R: Multiply High Signed×Unsigned Word (`R[d]: sext((R[s1][31:0] *su R[s2][31:0])[63:32])`) - RV64 only, upper 32 bits of 32-bit signed×unsigned multiply, sign-extended
    MULHSUW { d: Reg, s1: Reg, s2: Reg },
    /// R: Multiply High Unsigned Word (`R[d]: sext((R[s1][31:0] *u R[s2][31:0])[63:32])`) - RV64 only, upper 32 bits of 32-bit unsigned multiply, sign-extended
    MULHUW { d: Reg, s1: Reg, s2: Reg },

    //// RV64 A-extension Load-Reserved/Store-Conditional Instructions (64-bit) ////
    /// R: Load-Reserved Doubleword (`R[d]: M[R[s1]]`) - RV64 only, loads 64-bit doubleword, reserves address for SC
    LR_D { d: Reg, s1: Reg, aqrl: AqRl },
    /// R: Store-Conditional Doubleword (`R[d]: success/fail, M[R[s1]]: R[s2]`) - RV64 only, conditionally stores 64-bit doubleword if reservation valid
    SC_D { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },

    //// RV64 A-extension Atomic Memory Operations (64-bit) ////
    /// R: Atomic Add Doubleword (`R[d]: M[R[s1]], M[R[s1]]: M[R[s1]] + R[s2]`) - RV64 only, atomically adds s2 to 64-bit memory, returns original value
    AMOADD_D { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },
    /// R: Atomic Swap Doubleword (`R[d]: M[R[s1]], M[R[s1]]: R[s2]`) - RV64 only, atomically swaps s2 with 64-bit memory, returns original value
    AMOSWAP_D { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },
    /// R: Atomic AND Doubleword (`R[d]: M[R[s1]], M[R[s1]]: M[R[s1]] & R[s2]`) - RV64 only, atomically ANDs s2 with 64-bit memory, returns original value
    AMOAND_D { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },
    /// R: Atomic OR Doubleword (`R[d]: M[R[s1]], M[R[s1]]: M[R[s1]] | R[s2]`) - RV64 only, atomically ORs s2 with 64-bit memory, returns original value
    AMOOR_D { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },
    /// R: Atomic XOR Doubleword (`R[d]: M[R[s1]], M[R[s1]]: M[R[s1]] ^ R[s2]`) - RV64 only, atomically XORs s2 with 64-bit memory, returns original value
    AMOXOR_D { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },
    /// R: Atomic Max Doubleword (`R[d]: M[R[s1]], M[R[s1]]: max(M[R[s1]], R[s2])`) - RV64 only, atomically stores signed max with 64-bit memory, returns original value
    AMOMAX_D { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },
    /// R: Atomic Min Doubleword (`R[d]: M[R[s1]], M[R[s1]]: min(M[R[s1]], R[s2])`) - RV64 only, atomically stores signed min with 64-bit memory, returns original value
    AMOMIN_D { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },
    /// R: Atomic Max Unsigned Doubleword (`R[d]: M[R[s1]], M[R[s1]]: maxu(M[R[s1]], R[s2])`) - RV64 only, atomically stores unsigned max with 64-bit memory, returns original value
    AMOMAXU_D { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },
    /// R: Atomic Min Unsigned Doubleword (`R[d]: M[R[s1]], M[R[s1]]: minu(M[R[s1]], R[s2])`) - RV64 only, atomically stores unsigned min with 64-bit memory, returns original value
    AMOMINU_D { d: Reg, s1: Reg, s2: Reg, aqrl: AqRl },
}

impl I64 {
    #[inline]
    pub const fn into_u32(self) -> u32 {
        match self {
            // RV64I Base loads/stores
            LD { d, s, im }     => I32::i(0x03, d, 0b011, s, im),
            SD { s1, s2, im }   => I32::s(0x23, 0b011, s1, s2, im),
            LWU { d, s, im }    => I32::i(0x03, d, 0b110, s, im),

            // RV64I word operations
            ADDW { d, s1, s2 }  => I32::r(0x3B, d, 0b000, s1, s2, 0b0000000),
            SUBW { d, s1, s2 }  => I32::r(0x3B, d, 0b000, s1, s2, 0b0100000),
            ADDIW { d, s, im }  => I32::i(0x1B, d, 0b000, s, im),
            SLLW { d, s1, s2 }  => I32::r(0x3B, d, 0b001, s1, s2, 0b0000000),
            SRLW { d, s1, s2 }  => I32::r(0x3B, d, 0b101, s1, s2, 0b0000000),
            SRAW { d, s1, s2 }  => I32::r(0x3B, d, 0b101, s1, s2, 0b0100000),
            SLLIW { d, s, shamt } => I32::i7(0x1B, d, 0b001, s, shamt, 0b0000000),
            SRLIW { d, s, shamt } => I32::i7(0x1B, d, 0b101, s, shamt, 0b0000000),
            SRAIW { d, s, shamt } => I32::i7(0x1B, d, 0b101, s, shamt, 0b0100000),

            // RV64 M-extension word operations
            MULW { d, s1, s2 }    => I32::r(0x3B, d, 0b000, s1, s2, 0b0000001),
            DIVW { d, s1, s2 }    => I32::r(0x3B, d, 0b100, s1, s2, 0b0000001),
            DIVUW { d, s1, s2 }   => I32::r(0x3B, d, 0b101, s1, s2, 0b0000001),
            REMW { d, s1, s2 }    => I32::r(0x3B, d, 0b110, s1, s2, 0b0000001),
            REMUW { d, s1, s2 }   => I32::r(0x3B, d, 0b111, s1, s2, 0b0000001),
            MULHW { d, s1, s2 }   => I32::r(0x3B, d, 0b001, s1, s2, 0b0000001),
            MULHSUW { d, s1, s2 } => I32::r(0x3B, d, 0b010, s1, s2, 0b0000001),
            MULHUW { d, s1, s2 }  => I32::r(0x3B, d, 0b011, s1, s2, 0b0000001),

            // RV64 A-extension Load-Reserved/Store-Conditional (64-bit)
            LR_D { d, s1, aqrl }      => I32::amo(0x2F, d, 0b011, s1, ZERO, aqrl, 0b00010),
            SC_D { d, s1, s2, aqrl }  => I32::amo(0x2F, d, 0b011, s1, s2, aqrl, 0b00011),

            // RV64 A-extension Atomic Memory Operations (64-bit)
            AMOADD_D { d, s1, s2, aqrl }  => I32::amo(0x2F, d, 0b011, s1, s2, aqrl, 0b00000),
            AMOSWAP_D { d, s1, s2, aqrl } => I32::amo(0x2F, d, 0b011, s1, s2, aqrl, 0b00001),
            AMOAND_D { d, s1, s2, aqrl }  => I32::amo(0x2F, d, 0b011, s1, s2, aqrl, 0b01100),
            AMOOR_D { d, s1, s2, aqrl }   => I32::amo(0x2F, d, 0b011, s1, s2, aqrl, 0b01000),
            AMOXOR_D { d, s1, s2, aqrl }  => I32::amo(0x2F, d, 0b011, s1, s2, aqrl, 0b00100),
            AMOMAX_D { d, s1, s2, aqrl }  => I32::amo(0x2F, d, 0b011, s1, s2, aqrl, 0b10100),
            AMOMIN_D { d, s1, s2, aqrl }  => I32::amo(0x2F, d, 0b011, s1, s2, aqrl, 0b10000),
            AMOMAXU_D { d, s1, s2, aqrl } => I32::amo(0x2F, d, 0b011, s1, s2, aqrl, 0b11100),
            AMOMINU_D { d, s1, s2, aqrl } => I32::amo(0x2F, d, 0b011, s1, s2, aqrl, 0b11000),
        }
    }

    #[allow(clippy::match_single_binding)]
    pub const fn try_from_u32(with: u32) -> Result<Self, ConversionError> {
        Ok(match with & 0b1111111 {
            // Load From RAM (64-bit specific)
            0b0000011 => match I32::from_i(with) {
                (d, 0b011, s, im) => LD { d, s, im },
                (d, 0b110, s, im) => LWU { d, s, im },
                (_, funct, _, _) => {
                    return Err(ConversionError::UnknownFunct3(funct))
                }
            },
            // Immediate Operations (32-bit word operations) - OP-IMM-32
            0b0011011 => match I32::from_i(with) {
                (d, 0b000, s, im) => ADDIW { d, s, im },
                (d, 0b001, s, im) => match (im >> 5) & 0b1111111 {
                    0b0000000 => SLLIW { d, s, shamt: im as u8 },
                    _ => return Err(ConversionError::UnknownFunct7((im >> 5) as _))
                },
                (d, 0b101, s, im) => match (im >> 5) & 0b1111111 {
                    0b0000000 => SRLIW { d, s, shamt: im as u8 },
                    0b0100000 => SRAIW { d, s, shamt: im as u8 },
                    _ => return Err(ConversionError::UnknownFunct7((im >> 5) as _))
                },
                (_, funct, _, _) => {
                    return Err(ConversionError::UnknownFunct3(funct))
                }
            },
            // Store To RAM (64-bit specific)
            0b0100011 => match I32::from_s(with) {
                (0b011, s1, s2, im) => SD { s1, s2, im },
                (funct, _, _, _) => {
                    return Err(ConversionError::UnknownFunct3(funct))
                }
            },
            // AMO (Atomic Memory Operations) - A-extension (64-bit specific)
            0b0101111 => match I32::from_amo(with) {
                (d, 0b011, s1, _,  aqrl, 0b00010) => LR_D { d, s1, aqrl },
                (d, 0b011, s1, s2, aqrl, 0b00011) => SC_D { d, s1, s2, aqrl },
                (d, 0b011, s1, s2, aqrl, 0b00000) => AMOADD_D { d, s1, s2, aqrl },
                (d, 0b011, s1, s2, aqrl, 0b00001) => AMOSWAP_D { d, s1, s2, aqrl },
                (d, 0b011, s1, s2, aqrl, 0b01100) => AMOAND_D { d, s1, s2, aqrl },
                (d, 0b011, s1, s2, aqrl, 0b01000) => AMOOR_D { d, s1, s2, aqrl },
                (d, 0b011, s1, s2, aqrl, 0b00100) => AMOXOR_D { d, s1, s2, aqrl },
                (d, 0b011, s1, s2, aqrl, 0b10100) => AMOMAX_D { d, s1, s2, aqrl },
                (d, 0b011, s1, s2, aqrl, 0b10000) => AMOMIN_D { d, s1, s2, aqrl },
                (d, 0b011, s1, s2, aqrl, 0b11100) => AMOMAXU_D { d, s1, s2, aqrl },
                (d, 0b011, s1, s2, aqrl, 0b11000) => AMOMINU_D { d, s1, s2, aqrl },
                (_, funct3, _, _, _, funct5) => {
                    return Err(ConversionError::UnknownFunct3Funct5(funct3, funct5))
                }
            },
            // Register Operations (32-bit word operations) - OP-32
            0b0111011 => match I32::from_r(with) {
                (d, 0b000, s1, s2, 0b0000000) => ADDW { d, s1, s2 },
                (d, 0b000, s1, s2, 0b0100000) => SUBW { d, s1, s2 },
                (d, 0b001, s1, s2, 0b0000000) => SLLW { d, s1, s2 },
                (d, 0b101, s1, s2, 0b0000000) => SRLW { d, s1, s2 },
                (d, 0b101, s1, s2, 0b0100000) => SRAW { d, s1, s2 },
                // M-extension word operations
                (d, 0b000, s1, s2, 0b0000001) => MULW { d, s1, s2 },
                (d, 0b001, s1, s2, 0b0000001) => MULHW { d, s1, s2 },
                (d, 0b010, s1, s2, 0b0000001) => MULHSUW { d, s1, s2 },
                (d, 0b011, s1, s2, 0b0000001) => MULHUW { d, s1, s2 },
                (d, 0b100, s1, s2, 0b0000001) => DIVW { d, s1, s2 },
                (d, 0b101, s1, s2, 0b0000001) => DIVUW { d, s1, s2 },
                (d, 0b110, s1, s2, 0b0000001) => REMW { d, s1, s2 },
                (d, 0b111, s1, s2, 0b0000001) => REMUW { d, s1, s2 },
                (_, funct3, _, _, funct7) => {
                    return Err(ConversionError::UnknownFunct3Funct7(funct3, funct7))
                }
            },
            _ => return Err(ConversionError::UnknownOpcode(with & 0b1111111))
        })
    }
}

impl From<I64> for u32 {
    #[inline(always)]
    fn from(with: I64) -> Self {
        I64::into_u32(with)
    }
}

impl TryFrom<u32> for I64 {
    type Error = ConversionError;
    // Using match makes it easier to extend code in the future.
    #[allow(clippy::match_single_binding)]
    fn try_from(with: u32) -> Result<Self, Self::Error> {
        Self::try_from_u32(with)
    }
}

